from __future__ import annotations
import argparse
import functools

import datetime
from typing import Any
from typing import Sequence

import trio
import httpx
from types import SimpleNamespace
from typing import NamedTuple
from pydantic import BaseModel
import logging
from pythonjsonlogger import jsonlogger
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import wait_random_exponential
from tenacity import stop_after_delay
import tqdm
import duckdb
from trio.lowlevel import Task
from fsspec.implementations.memory import MemoryFileSystem
from typing import Any


API_URL = "https://api.finra.org"
API_LIMIT = 5000

LOG = logging.getLogger(__name__)
MEM_FS = MemoryFileSystem()
PROGRESSBAR = False


class RateLimitedClient(httpx.AsyncClient):
    def __init__(self, rate: float, **kwargs: Any) -> None:
        self.rate = rate
        self.next_allowed_time = trio.current_time()
        self.log = logging.getLogger(self.__class__.__name__)
        super().__init__(**kwargs)

    async def send(self, *args: Any, **kwargs: Any) -> httpx.Response:
        while (now := trio.current_time()) < self.next_allowed_time:
            self.log.debug(
                "RateLimitedClient waiting",
                extra={
                    "rate": self.rate,
                    "current_time": now,
                    "next_allowed_time": self.next_allowed_time,
                },
            )
            await trio.sleep(self.next_allowed_time - trio.current_time())

        self.next_allowed_time = trio.current_time() + 1 / self.rate
        return await super().send(*args, **kwargs)


@functools.total_ordering
class DateRange:
    def __init__(self, start: datetime.date, end: datetime.date) -> None:
        if start > end:
            raise ValueError(f"{start=} must be before {end=}")
        self.start = start
        self.end = end

    @functools.cached_property
    def delta(self) -> datetime.timedelta:
        return self.end - self.start

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DateRange):
            return NotImplemented
        return (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, (DateRange, datetime.date, datetime.datetime)):
            return NotImplemented
        elif isinstance(other, DateRange):
            return self.start < other.start
        elif isinstance(other, datetime.datetime):
            return self.start < other.date()
        else:
            return self.start < other

    def __repr__(self) -> str:
        return f"DateRange(start={self.start!r}, end={self.end!r})"


class Field(BaseModel):
    name: str
    type: str
    description: str


class Metadata(BaseModel):
    datasetGroup: str
    datasetName: str
    description: str
    partitionFields: list[str]
    fields: list[Field]

    def _validate_fields(self, fields: list[str] | None) -> list[str]:
        all_fields = {f.name for f in self.fields}

        if fields is None:
            return list(all_fields)

        unsupported = set(fields).difference(all_fields)
        if unsupported:
            raise ValueError(f"Unsupported fields: {unsupported}")

        return fields

    def _date_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.type == "Date"]


class Partition(BaseModel):
    partitions: list[str]


class Partitions(BaseModel):
    datasetGroup: str
    datasetName: str
    partitionFields: list[str]
    availablePartitions: list[Partition]


class DatasetInfo(SimpleNamespace):
    metadata: Metadata
    partitions: Partitions

    def create_table(self, fields: list[str]) -> None:
        type_map = {
            "Date": "DATE",
            "String": "STRING",
            "Number": "DOUBLE",
        }
        field_types = ", ".join(
            f"{f.name} {type_map[f.type]}"
            for f in self.metadata.fields
            if f.name in fields
        )
        duckdb.sql(f"CREATE SCHEMA {self.metadata.datasetGroup}")
        duckdb.sql(
            f"CREATE TABLE {self.metadata.datasetGroup}.{self.metadata.datasetName} ({field_types})"
        )

    def validate_fields(self, fields: list[str] | None) -> list[str]:
        return self.metadata._validate_fields(fields)

    def date_fields(self) -> list[str]:
        return self.metadata._date_fields()


class Dataset(NamedTuple):
    group: str
    name: str


@retry(
    retry=retry_if_exception_type(httpx.ReadTimeout),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_delay(120),
    sleep=trio.sleep,
)
async def get_metadata(
    dataset: Dataset, info: DatasetInfo, *, client: httpx.AsyncClient
) -> None:
    LOG.debug("starting load metadata", extra={"dataset": dataset})
    response = await client.get(
        f"{API_URL}/metadata/group/{dataset.group}/name/{dataset.name}"
    )
    response.raise_for_status()
    info.metadata = Metadata.model_validate(response.json())
    LOG.debug("finished load metadata", extra={"dataset": dataset})


@retry(
    retry=retry_if_exception_type(httpx.ReadTimeout),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_delay(120),
    sleep=trio.sleep,
)
async def get_partitions(
    dataset: Dataset, info: DatasetInfo, *, client: httpx.AsyncClient
) -> None:
    LOG.debug("starting load partitions", extra={"dataset": dataset})
    response = await client.get(
        f"{API_URL}/partitions/group/{dataset.group}/name/{dataset.name}"
    )
    response.raise_for_status()
    info.partitions = Partitions.model_validate(response.json())
    LOG.debug("finished load partitions", extra={"dataset": dataset})


async def load_data(
    dataset: Dataset,
    fields: list[str],
    dates: DateRange | None,
    *,
    client: httpx.AsyncClient,
) -> None:
    num_records = await _get_total_records(dataset, fields, dates, client=client)

    offsets = list(range(0, num_records, API_LIMIT))
    if PROGRESSBAR:
        instrument = TrioProgress(len(offsets))
        trio.lowlevel.add_instrument(instrument)
    async with trio.open_nursery() as nursery:
        for offset in offsets:
            nursery.start_soon(
                functools.partial(
                    _get_data,
                    dataset,
                    fields,
                    dates,
                    offset=offset,
                    client=client,
                )
            )
    if PROGRESSBAR:
        trio.lowlevel.remove_instrument(instrument)


async def _get_total_records(
    dataset: Dataset,
    fields: list[str],
    dates: DateRange | None,
    *,
    client: httpx.AsyncClient,
) -> int:
    response = await _query_data(
        dataset, fields, dates, offset=0, client=client, limit=1
    )
    return int(response.headers["record-total"])


async def _get_data(
    dataset: Dataset,
    fields: list[str],
    dates: DateRange | None,
    *,
    offset: int,
    client: httpx.AsyncClient,
) -> None:
    LOG.debug(
        "starting load data",
        extra={"dataset": dataset, "fields": fields, "offset": offset, "dates": dates},
    )
    response = await _query_data(dataset, fields, dates, offset=offset, client=client)
    file_id = response.headers["finra-api-request-id"]
    MEM_FS.write_text(path=f"{file_id}", value=response.text)
    duckdb.sql(
        f"INSERT INTO {dataset.group}.{dataset.name} BY NAME SELECT * FROM read_csv('memory:///{file_id}')"
    )
    LOG.debug(
        "finished load data",
        extra={"dataset": dataset, "fields": fields, "offset": offset, "dates": dates},
    )


@retry(
    retry=retry_if_exception_type(httpx.ReadTimeout),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_delay(120),
    sleep=trio.sleep,
)
async def _query_data(
    dataset: Dataset,
    fields: list[str],
    dates: DateRange | None,
    *,
    offset: int,
    client: httpx.AsyncClient,
    limit: int = API_LIMIT,
) -> httpx.Response:
    payload = {
        "fields": fields,
        "limit": limit,
        "offset": offset,
    }

    if dates is not None:
        payload["dateRangeFilters"] = [
            {
                "startDate": dates.start.isoformat(),
                "endDate": dates.end.isoformat(),
                # TODO: figure out how to make this more dynamic
                "fieldName": "tradeReportDate",
            }
        ]
    response = await client.post(
        f"{API_URL}/data/group/{dataset.group}/name/{dataset.name}",
        json=payload,
    )
    response.raise_for_status()
    return response


class TrioProgress(trio.abc.Instrument):
    def __init__(self, total: int) -> None:
        self.tqdm = tqdm.tqdm(total=total)

    def task_exited(self, task: Task) -> None:
        self.tqdm.update(1)


async def amain(
    group: str, name: str, fields: list[str] | None, dates: DateRange | None
) -> int:
    dataset = Dataset(group, name)
    info = DatasetInfo()
    # FINRA requests rate limit of 20 requests per second so we do 10 to have a decent buffer
    async with RateLimitedClient(rate=10) as client:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                functools.partial(get_metadata, dataset, info, client=client)
            )
            nursery.start_soon(
                functools.partial(get_partitions, dataset, info, client=client)
            )

        fields = info.validate_fields(fields)
        info.create_table(fields)
        await load_data(dataset, fields, dates, client=client)

    duckdb.table(f"{dataset.group}.{dataset.name}").show()
    print(duckdb.table(f"{dataset.group}.{dataset.name}").count("*"))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--group", type=str, default="OTCMarket")
    parser.add_argument("-n", "--name", type=str, default="regShoDaily")
    parser.add_argument(
        "-s",
        "--start",
        type=datetime.date.fromisoformat,
        default=datetime.date.today().replace(month=1, day=1),
    )
    parser.add_argument(
        "-e", "--end", type=datetime.date.fromisoformat, default=datetime.date.today()
    )
    parser.add_argument("-f", "--fields", type=str, nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args(argv)

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)

    LOG.addHandler(handler)

    if args.show_progress:
        global PROGRESSBAR
        PROGRESSBAR = True

    if args.verbose == 0:
        LOG.setLevel(logging.INFO)
    else:
        LOG.setLevel(logging.DEBUG)

    if args.verbose > 1:
        logging.getLogger("RateLimitedClient").setLevel(logging.DEBUG)
        logging.getLogger("RateLimitedClient").addHandler(handler)

    dates = DateRange(args.start, args.end)

    duckdb.register_filesystem(MEM_FS)
    trio.run(amain, args.group, args.name, args.fields, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
