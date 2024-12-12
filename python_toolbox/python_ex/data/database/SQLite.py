from __future__ import annotations

from typing import Any, Literal, TypeVar

import json

from dataclasses import InitVar, dataclass, field
import sqlite3

from pathlib import Path
from python_ex.project import Config as CFG


class Data():
    @staticmethod
    def Flag_to_type(flag: str):
        if flag == "INTEGER":
            return int
        if flag == "REAL":
            return float
        if flag == "TEXT":
            return str
        return None  # BLOB


class Config():
    cfg = TypeVar("cfg", bound=CFG.Basement)

    @dataclass
    class Column(CFG.Basement):
        data_sqlite_type: Literal["INTEGER", "REAL", "TEXT", "BLOB"]

        # option for this column
        empty_able: bool = True
        is_unique: bool = False
        default_value: Any = None

        # option for when this column data type is text
        collate: str = "BINARY"
        is_time: bool = False

        # option for relation that with in same row or the other table
        primary: bool = False
        relation: tuple[str, str] | None = None

        skip_able: bool = field(init=False)

        def __post_init__(self):
            self.skip_able = any([
                self.default_value is not None,
                self.empty_able,
                (self.primary and self.data_sqlite_type == "INTEGER")
            ])

        def Value_check(self, value: Any):
            return value  # this code is not apply value check.

        def Set_default_value(self, value: Any, is_time: bool = False):
            _c_type = self.data_sqlite_type
            _type = Data.Flag_to_type(_c_type)

            if is_time and self.is_time:
                # in later add to function for time class change to int or text
                # if isinstance(value, time):
                #     value = value
                ...

            if _type is None or isinstance(value, _type):
                self.default_value = self.Value_check(value)
                return 0, "done"

            return 1, f"Type mismatch, {_c_type}(={_type}) != {type(value)}"

        def Ref_command(self, name: str):
            _rel = self.relation

            if _rel is not None:
                return f"FOREIGN KEY ({name}) REFERENCES {_rel[0]}({_rel[1]})"
            return ""

        def Create_command(self, name: str):
            _opt = [f"\t{name}", self.data_sqlite_type]

            if not self.empty_able:
                _opt.append("NOT NULL")
            if self.is_unique:
                _opt.append("UNIQUE")
            if self.relation is not None:
                _rel = self.relation
                _opt.append(f"REFERENCES {_rel[0]}({_rel[1]})")
            if self.default_value is not None:
                _opt.append(f"DEFAULT {self.default_value}")

            return " ".join(_opt)

        def Insert_command(self, value: Any):
            if isinstance(value, str):
                return f"'{value}'"

            # if isinstance(value, time):
            #     return ""

            return str(value)

    @dataclass
    class Table(CFG.Basement):
        column_params: InitVar[dict[str, dict[str, Any] | Config.Column]]
        column_cfgs: dict[str, Config.Column] = field(init=False)

        is_without_row_id: bool = False

        def __post_init__(
            self, column_params: dict[str, dict[str, Any] | Config.Column]
        ):
            self.column_cfgs = dict((
                _n, Config.Prams_to_config(_p, Config.Column)
            ) for _n, _p in column_params.items())

        def Config_to_dict(self) -> dict[str, Any]:
            return {
                "column_params": dict((
                    _name,
                    _cfg.Config_to_dict()
                ) for _name, _cfg in self.column_cfgs.items()),
                "is_without_row_id": self.is_without_row_id
            }

        def Create_command(self, name: str):
            _c_cmd_list: list[str] = []
            _primary_list: list[str] = []
            _relation_list: list[str] = []

            for _c_name, _c_cfg in self.column_cfgs.items():
                _c_cmd: str = _c_cfg.Create_command(_c_name)
                _c_cmd_list.append(f"{_c_cmd}")

                if _c_cfg.primary:
                    _primary_list.append(_c_name)

                if _c_cfg.relation is not None:
                    _cons = f"{name}_FK_{_c_name}"
                    _relation_list.append(
                        f"\tCONSTRAINT {_cons} {_c_cfg.Ref_command(_c_name)}",
                    )

            _p_keys = ", ".join(_primary_list)
            _primary_cmd = f"\tCONSTRAINT {name}_PK PRIMARY KEY ({_p_keys})"

            return 0, "\n".join([
                " ".join(["CREATE", "TABLE", f"{name} ("]),
                ",\n".join(_c_cmd_list + [_primary_cmd] + _relation_list),
                f"){'WITHOUT ROWID' if self.is_without_row_id else ''};"
            ])

        def Insert_command(self, name: str, values: dict[str, Any]):
            _cfgs = self.column_cfgs
            _c_names = []
            _c_values = []
            for _c_name, _c_cfg in _cfgs.items():
                if _c_name not in values:
                    if _c_cfg.skip_able:
                        continue  # Set default value

                    return (
                        1,
                        " ".join([
                            "There is no input data for",
                            f"column '{_c_name}' in table '{name}',"
                            "which has no default value and not apply NULL."])
                    )

                _insert_v = values[_c_name]

                if not _c_cfg.Value_check(_insert_v):
                    return (
                        2,
                        ""
                    )

                _c_names.append(_c_name)
                _c_values.append(_c_cfg.Insert_command(_insert_v))

            _n_kw = ", ".join(_c_names)
            _v = ", ".join(_c_values)

            return (0, f"INSERT INTO {name} ({_n_kw}) VALUES ({_v});")

    @dataclass
    class Database(CFG.Basement):
        save_dir: str
        name: str

        table_params: InitVar[dict[str, dict[str, Any] | Config.Table]]
        table_cfgs: dict[str, Config.Table] = field(init=False)

        def __post_init__(
            self, table_params: dict[str, dict[str, Any] | Config.Table]
        ):
            self.table_cfgs = dict((
                _n, Config.Prams_to_config(_p, Config.Table)
            ) for _n, _p in table_params.items())

        def Config_to_dict(self) -> dict[str, Any]:
            return {
                "save_dir": self.save_dir,
                "name": self.name,
                "table_params": dict((
                    _name, _cfg.Config_to_dict()
                ) for _name, _cfg in self.table_cfgs.items())
            }

    @staticmethod
    def Prams_to_config(
        param: dict[str, Any] | cfg, cfg_type: type[cfg]
    ) -> cfg:
        return cfg_type(**param) if isinstance(param, dict) else param

    @staticmethod
    def Load_from_file(name: str, save_dir: str | None = None):
        _path = (Path().cwd() if save_dir is None else Path(save_dir)) / Path(name)

        with _path.open(encoding="UTF-8") as cfg_file:
            if _path.suffix == "json":
                return Config.Database(**json.load(cfg_file))

        raise ValueError()


class Database():
    def __init__(
        self,
        name: str,
        tables: dict[str, dict[str, Any] | Config.Table],
        save_dir: str | None = None
    ):
        self.db_path = (Path().cwd() if save_dir is None else Path(save_dir)) / Path(name)
        self.table_cfgs: dict[str, Config.Table] = {}

        self.db: sqlite3.Connection = sqlite3.connect(self.db_path)

        for _t_name, _table_params in tables.items():
            if isinstance(_table_params, Config.Table):
                if self.Check(_t_name):
                    self.table_cfgs[_t_name] = _table_params
                    continue
                self.Apply(*_table_params.Create_command(_t_name))
                self.table_cfgs[_t_name] = _table_params
            else:
                try:
                    self.Create_table(_t_name, **_table_params)
                except ValueError:
                    ...

    def Save_to_cfg_file(self, name: str, save_dir: str | None = None):
        _cfg_path = (Path().cwd() if save_dir is None else Path(save_dir)) / Path(name)

        _db_cfg = Config.Database(str(self.db_path.parent), self.db_path.name, {})
        _db_cfg.table_cfgs = self.table_cfgs

        with _cfg_path.open("w", encoding="UTF-8") as cfg_file:
            if "json" in _cfg_path.suffix:
                json.dump(_db_cfg.Config_to_dict(), cfg_file, indent=4)

    def Apply(self, error_code: int, cmd: str):
        if error_code:
            raise ValueError(cmd)

        _cs = self.db.cursor()
        _cs.execute(cmd)
        return _cs

    # def Commit(self):
    #     self.Commit()

    def Disconnect(self):
        self.db.close()

    def Check(self, name: str) -> bool:
        _cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        _cs = self.Apply(0, _cmd)
        for _table_name in _cs.fetchall():
            if name in _table_name:
                return True
        return False

    def Copy(self, name: str):
        ...

    def Drop_table(self, name: str):
        raise NotImplementedError
        # _cs = self.db.cursor()
        # _cs.execute(f"DROP TABLE IF EXISTS {name};")

    def Create_table(
        self, name: str,
        column_params: dict[str, Config.Column | dict[str, Any]],
        is_without_row_id: bool = False,
        is_override: bool = False
    ):
        # check the table "name" is exist in db
        if self.Check(name):
            if not is_override:
                self.Apply(1, f"Table '{name}' is already exist.")

            # _legacy = self.Drop_table(name)

            # del the legacy table in db

        _t_cfg = Config.Table(column_params, is_without_row_id)
        self.Apply(*_t_cfg.Create_command(name))

        # if is_override:
        #     _legacy

        self.table_cfgs[name] = _t_cfg

    def Insert_value(self, name: str, value: dict[str, Any]):
        self.Apply(*self.table_cfgs[name].Insert_command(name, value))
        self.db.commit()
