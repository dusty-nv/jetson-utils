#!/usr/bin/env python3
import argparse

from jetson_utils import getLogger, Env, String

log = getLogger(__name__)

__all__ = ['Test']

class Test:
    @staticmethod
    def functions(**kwargs):
        return dict(
            logging = Test.logging,
            env = Test.env,
            table = Test.table
        )

    @staticmethod
    def all(**kwargs):
        for k,v in Test.functions(**kwargs).items():
            log.warning(f"RUNNING jetson_utils.test.{k}()\n")
            v(**kwargs)
            print('')
            log.success(f"✅ DONE jetson_utils.test.{k}()\n")

    @staticmethod
    def logging(msg: str=None, **kwargs):
        if not msg:
            msg = "Testing abc 123"

        for k,v in log.getLevels().items():
            log.log(v, f"log.log({k.upper()}) - {msg}")

        print('')

        for k,v in log.getLevels().items():
            getattr(log, k)(f"log.{k}() - {msg}")

    @staticmethod
    def env(**kwargs):
        print(Env)

    @staticmethod
    def table(**kwargs):
        print(String.table(Env))

if __name__ == '__main__':
    Test.all()