#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

championmasteryapi.get_champion_mastery_score = auto_retry(championmasteryapi.get_champion_mastery_score)

def main():
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", db="lol")
    cursor = db.cursor()
    db.autocommit(True)  # Autocommit INSERTs to spotify DB
    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    # Setup riotapi
    riotapi.set_region("NA")
    riotapi.print_calls(True)
    key = os.environ["DEV_KEY2"]  # You can create an env var called "DEV_KEY" that holds your developer key. It will be loaded here.
    riotapi.set_api_key(key)
    riotapi.set_load_policy(LoadPolicy.lazy)

    # Get total masteries
    cursor.execute("SELECT id FROM Summoner")
    summoners = list(cursor)
    for (summoner,) in summoners:
        cursor.execute("SELECT EXISTS (SELECT * FROM SummonerMasteries WHERE summId = %s)", [summoner])
        is_present = list(cursor)[0][0]
        if not is_present:
            cursor.execute("SELECT region "
                           "FROM MatchParticipant PA, MatchPlayer PL, MatchDetail D "
                           "WHERE PL.summonerId = %s "
                           "AND PL._participant_id = PA._id "
                           "AND PA._match_id = D.matchId", [summoner])
            region = list(cursor)
            if region:
                region = region[0][0]
                riotapi.set_region(region)
                mastery_score = championmasteryapi.get_champion_mastery_score(summoner)
            else:
                mastery_score = 0
            cursor.execute("INSERT INTO SummonerMasteries (summId, mastery) VALUES (%s, %s)", (summoner, mastery_score))

    cursor.close()
    db.close()


if __name__ == "__main__":
    main()