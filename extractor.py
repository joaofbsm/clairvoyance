from __future__ import print_function

import os
import sys
import MySQLdb
import pandas as pd
import numpy as np

def onehot_team_champions(match, db):
    champion = pd.read_sql("SELECT id FROM Champion", db)
    blue_champions = np.zeros(champion.shape[0], dtype="int")
    red_champions = np.zeros(champion.shape[0], dtype="int")
    winner = np.array(match["winner"].iloc[0], dtype="int")[np.newaxis]
    for i, value in match[:5].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        blue_champions[index] = 1

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1

    instance = np.concatenate((blue_champions, red_champions, winner))
    return instance


def onehot_team_champions_spells(match, db):
    champion = pd.read_sql("SELECT id FROM Champion", db)
    blue_champions = np.zeros(champion.shape[0], dtype="int")
    red_champions = np.zeros(champion.shape[0], dtype="int")

    spell = pd.read_sql("SELECT id FROM SummonerSpell "
                        "WHERE name='Barrier' OR name='Cleanse' "
                        "OR name='Exhaust' OR name='Flash' OR name='Ghost' "
                        "OR name='Heal' OR name='Ignite' OR name='Smite' "
                        "OR name='Teleport'", db)
    blue_spells = np.zeros(spell.shape[0], dtype="int")
    red_spells = np.zeros(spell.shape[0], dtype="int")

    winner = np.array(match["winner"].iloc[0], dtype="int")[np.newaxis]
    for i, value in match[:5].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        blue_champions[index] = 1

        index = spell.id[spell.id == value["spell1Id"]].index.tolist()[0]
        blue_spells[index] += 1
        index = spell.id[spell.id == value["spell2Id"]].index.tolist()[0]
        blue_spells[index] += 1

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1

        index = spell.id[spell.id == value["spell1Id"]].index.tolist()[0]
        red_spells[index] += 1
        index = spell.id[spell.id == value["spell2Id"]].index.tolist()[0]
        red_spells[index] += 1

    instance = np.concatenate((blue_spells, blue_champions, red_spells, red_champions, winner))
    return instance


def onehot_team_damage(match, champion_damage, db):
    champion = pd.read_sql("SELECT id FROM Champion", db)
    blue_champions = np.zeros(champion.shape[0], dtype="int")
    red_champions = np.zeros(champion.shape[0], dtype="int")
    blueteam_damage = np.zeros((3))
    redteam_damage = np.zeros((3))
    winner = np.array(match["winner"].iloc[0], dtype="int")[np.newaxis]
    for i, value in match[:5].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        blue_champions[index] = 1
        blueteam_damage += champion_damage[value["championId"]]

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1
        redteam_damage += champion_damage[value["championId"]]

    instance = np.concatenate((blue_champions, blueteam_damage, red_champions, redteam_damage, winner))
    return instance

def onehot_damage_percent(match, champion_damage, db):
    champion = pd.read_sql("SELECT id FROM Champion", db)
    blue_champions = np.zeros(champion.shape[0], dtype="int")
    red_champions = np.zeros(champion.shape[0], dtype="int")
    blueteam_damage = np.zeros((2))
    redteam_damage = np.zeros((2))
    winner = np.array(match["winner"].iloc[0], dtype="int")[np.newaxis]
    for i, value in match[:5].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        blue_champions[index] = 1
        blueteam_damage += champion_damage[value["championId"]]

    total_damage = np.sum(blueteam_damage)
    blueteam_damage = np.around(np.divide(blueteam_damage, total_damage), decimals=5)

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1
        redteam_damage += champion_damage[value["championId"]]

    total_damage = np.sum(redteam_damage)
    redteam_damage = np.around(np.divide(redteam_damage, total_damage), decimals=5)

    instance = np.concatenate((blue_champions, blueteam_damage, red_champions, redteam_damage, winner))
    #print(instance)
    return instance  


def build_model_pre1(db, cursor):
    df = pd.read_sql("SELECT D.matchId, P.championId, P.teamId, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 AND "
                     "D.matchId = T._match_id AND P.teamId = T.teamId "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 273), dtype="int")
    i = 0
    for match in xrange(10, df.shape[0], 10):
        print(match)
        dataset[i] = onehot_team_champions(df[match-10:match], db)
        #dataset[i] = np.concatenate((dataset[i], winner))
        #print(df.iloc[match - 10]["matchId"])
        i += 1

    np.savetxt("pre1.csv", dataset, delimiter=",", fmt="%i")


def build_model_pre2(db, cursor):
    df = pd.read_sql("SELECT D.matchId, P.championId, P.teamId, P.spell1Id, "
                     "P.spell2Id, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 AND "
                     "D.matchId = T._match_id AND P.teamId = T.teamId "
                     "ORDER BY D.matchId, P.teamId", db)

    dataset = np.zeros((df.shape[0] / 10, 291), dtype="int")
    i = 0
    for match in xrange(10, df.shape[0], 10):
        print(match)
        dataset[i] = onehot_team_champions_spells(df[match-10:match], db)
        i += 1

    np.savetxt("pre2.csv", dataset, delimiter=",", fmt="%i")


def build_model_pre5(db, cursor):
    df = pd.read_sql("SELECT D.matchId, P.championId, P.teamId, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 AND "
                     "D.matchId = T._match_id AND P.teamId = T.teamId "
                     "ORDER BY D.matchId, P.teamId ", db)

    dfd = pd.read_sql("SELECT _champion_id, attack, defense, magic "
                     "FROM ChampionInfo "
                     "ORDER BY _champion_id", db)
    champion_damage = dfd.set_index("_champion_id").T.to_dict("list")

    dataset = np.zeros((df.shape[0] / 10, 279), dtype="int")
    i = 0
    for match in xrange(10, df.shape[0], 10):
        print(match)
        dataset[i] = onehot_team_damage(df[match-10:match], champion_damage, db)
        i += 1

    np.savetxt("pre5.csv", dataset, delimiter=",", fmt="%i")

def build_model_pre6(db, cursor):
    df = pd.read_sql("SELECT D.matchId, P.championId, P.teamId, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 AND "
                     "D.matchId = T._match_id AND P.teamId = T.teamId "
                     "ORDER BY D.matchId, P.teamId ", db)

    dfd = pd.read_sql("SELECT _champion_id, attack, magic "
                     "FROM ChampionInfo "
                     "ORDER BY _champion_id", db)
    champion_damage = dfd.set_index("_champion_id").T.to_dict("list")

    dataset = np.zeros((df.shape[0] / 10, 277))
    i = 0
    for match in xrange(10, df.shape[0], 10):
        print(match)
        dataset[i] = onehot_damage_percent(df[match-10:match], champion_damage, db)
        #print(dataset[i])
        i += 1

    np.savetxt("pre6.csv", dataset, delimiter=",", fmt="%.5g")

def main(args):
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", db="lol")
    cursor = db.cursor()
    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    feature_models = {"pre1": build_model_pre1,
                      "pre2": build_model_pre2,
                      "pre5": build_model_pre5,
                      "pre6": build_model_pre6}
    model = feature_models[args[0]](db, cursor)

    cursor.close()
    db.close()


if __name__ == "__main__":
    main(sys.argv[1:])