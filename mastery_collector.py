import os
import MySQLdb
from datetime import datetime
from collections import deque

from cassiopeia import riotapi
from cassiopeia.dto import championmasteryapi
from cassiopeia.type.api.exception import APIError
from cassiopeia.type.core.common import LoadPolicy
from cassiopeia.type.api.store import SQLAlchemyDB

def auto_retry(api_call_method):
    """ A decorator to automatically retry 500s (Service Unavailable) and skip 400s (Bad Request) or 404s (Not Found). """
    def call_wrapper(*args, **kwargs):
        try:
            return api_call_method(*args, **kwargs)
        except APIError as error:
            # Try Again Once
            if error.error_code in [500, 503]:
                try:
                    print("Got a 500, trying again...")
                    return api_call_method(*args, **kwargs)
                except APIError as another_error:
                    if another_error.error_code in [500, 503, 400, 404]:
                        pass
                    else:
                        raise another_error

            # Skip
            elif error.error_code in [400, 404]:
                print("Got a 400 or 404")
                pass

            # Fatal
            else:
                raise error
    return call_wrapper


def main():
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", db="lol")
    cursor = db.cursor()
    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    cursor.execute("SELECT id FROM Summoner")
    summoners = list(cursor)
    id_ = summoners[0][0]

    cursor.close()
    db.close()

    # Setup riotapi
    riotapi.set_region("NA")
    riotapi.print_calls(True)
    key = os.environ["DEV_KEY"]  # You can create an env var called "DEV_KEY" that holds your developer key. It will be loaded here.
    riotapi.set_api_key(key)
    riotapi.set_load_policy(LoadPolicy.lazy)

    summoner = riotapi.get_summoner_by_id(id_)
    mast = championmasteryapi.get_champion_mastery_score(id_)
    print(mast)
    mast = riotapi.get_champion_mastery_score(summoner)
    print(mast)


if __name__ == "__main__":
    main()