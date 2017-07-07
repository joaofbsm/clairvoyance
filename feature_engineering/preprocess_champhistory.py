#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from tqdm import tqdm
import os
import MySQLdb

def main():
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", db="lol")
    cursor = db.cursor()
    db.autocommit(True)  # Autocommit INSERTs to spotify DB
    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    get_outcomes = ("SELECT T.winner, count(*) "
                    "FROM MatchParticipant P, MatchPlayer PL, MatchTeam T "
                    "WHERE PL.summonerId = %s "
                    "AND P.championId = %s "
                    "AND P._id = PL._participant_id AND P._match_id = T._match_id "
                    "AND P.teamId = T.teamId "
                    "GROUP BY T.winner "
                    "ORDER BY T.winner")

    insert_outcomes = ("INSERT INTO "
                       "SummonerChampHistory (summId, championId, wins, losses, rate) "
                       "VALUES (%s, %s, %s, %s, %s)")

    cursor.execute("SELECT summonerId, championId FROM MatchParticipant PA, MatchPlayer PL WHERE PL._participant_id = PA._id")
    result = list(cursor)

    bar = tqdm(total=len(result))
    for summoner, champion in result:
        cursor.execute("SELECT EXISTS (SELECT * FROM SummonerChampHistory WHERE summId = %s AND championId = %s)", (summoner, champion))
        is_present = list(cursor)[0][0]
        if not is_present:
            wins = 0
            losses = 0
            cursor.execute(get_outcomes, (summoner, champion))
            outcomes = list(cursor)
            if not outcomes:
                bar.update(1)
                continue
            elif len(outcomes) == 2:
                losses = outcomes[0][1]
                wins = outcomes[1][1]
            else:
                if outcomes[0][0] == 0:
                    losses = outcomes[0][1]
                else:
                    wins = outcomes[0][1]
            total = wins + losses
            rate = (wins / total) * 100

            cursor.execute(insert_outcomes, (summoner, champion, wins, losses, rate))
            
        bar.update(1)

    cursor.close()
    db.close()



if __name__ == "__main__":
    main()