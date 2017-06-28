#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Feature extractor and model builder"""

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


def onehot_team_masteries(match, masteries, db):
    champion = pd.read_sql("SELECT id FROM Champion", db)
    blue_champions = np.zeros(champion.shape[0], dtype="int")
    red_champions = np.zeros(champion.shape[0], dtype="int")
    winner = np.array(match["winner"].iloc[0], dtype="int")[np.newaxis]

    gb = masteries.groupby(["matchId", "teamId"])
    for name, group in gb:
        print(name)
        print(group)

    for i, value in match[:5].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        blue_champions[index] = 1

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1

    instance = np.concatenate((blue_champions, red_champions, winner))
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
    blueteam_damage = np.around(np.divide(blueteam_damage, total_damage), decimals=5) * 100

    for i, value in match[5:10].iterrows():
        index = champion.id[champion.id == value["championId"]].index.tolist()[0]
        red_champions[index] = 1
        redteam_damage += champion_damage[value["championId"]]

    total_damage = np.sum(redteam_damage)
    redteam_damage = np.around(np.divide(redteam_damage, total_damage), decimals=5) * 100

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


def build_model_pre3(db, cursor):

    df = pd.read_sql("SELECT D.matchId, P.championId, P.teamId, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 AND "
                     "D.matchId = T._match_id AND P.teamId = T.teamId "
                     "ORDER BY D.matchId, P.teamId ", db)

    dfm = pd.read_sql("SELECT D.matchId, P.teamId, M.masteryId, sum(M.rank) "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, MatchMastery M "
                     "WHERE P._match_id = D.matchId AND D.matchId = T._match_id "
                     "AND P.teamId = T.teamId AND P._id = M._participant_id AND D.mapId = 11 "
                     "GROUP BY D.matchId, P.teamId, M.masteryId", db)

    gb = dfm.groupby(["matchId"])
    dataset = np.zeros((df.shape[0] / 10, 363), dtype="int")
    i = 0
    for match in xrange(10, df.shape[0], 10):
        print(match)
        print(str(df[match-10:match-9]["matchId"].item()))
        dataset[i] = onehot_team_masteries(df[match-10:match], gb["matchId"].get_group(str(df[match-10:match-9]["matchId"].item())), db)
        #dataset[i] = np.concatenate((dataset[i], winner))
        #print(df.iloc[match - 10]["matchId"])
        i += 1

    np.savetxt("pre1.csv", dataset, delimiter=",", fmt="%i")
        

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


def onehot_champions(match, db):
    champions = pd.read_sql("SELECT id FROM Champion", db)
    champions["pos"] = champions.index
    champions = champions.set_index("id").to_dict()

    blue_team = match[:5]
    red_team = match[5:10]

    blue_champions = np.zeros(len(champions["pos"]), dtype="int")
    red_champions = np.zeros(len(champions["pos"]), dtype="int")

    for _, player in blue_team.iterrows():
        pos = champions["pos"][player["championId"]]
        blue_champions[pos] = 1

    for _, player in red_team.iterrows():
        pos = champions["pos"][player["championId"]]
        red_champions[pos] = 1

    result = np.concatenate((blue_champions, red_champions))
    return result


def mastery_scores_team(match, cursor):
    get_mastery_scores = ("SELECT mastery "
                          "FROM SummonerMasteries "
                          "WHERE summId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    mastery_scores = np.zeros(2, dtype="int")
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)[0][0]
        mastery_scores[0] += mastery_score


    for _, player in red_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)[0][0]
        mastery_scores[1] += mastery_score

    return mastery_scores


def champion_masteries_team(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    champion_masteries = np.zeros(2, dtype="int")
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        champion_masteries[0] += champion_mastery


    for _, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        champion_masteries[1] += champion_mastery

    return champion_masteries


def champion_masteries_summoner(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_champion_masteries = np.zeros(5, dtype="int")
    red_champion_masteries = np.zeros(5, dtype="int")
    
    for i, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        blue_champion_masteries[i] = champion_mastery


    for i, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        red_champion_masteries[i] = champion_mastery

    champion_masteries = np.concatenate((blue_champion_masteries, 
                                         red_champion_masteries))

    return champion_masteries

def build_model_pre7(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 275))
    for i, match in enumerate(xrange(0, df.shape[0] - 10, 10)):
        print(i + 1)  # Current processing match

        champions = onehot_champions(df[match:match + 10], db)
        mastery_scores = mastery_scores_team(df[match:match + 10], cursor)
        #mastery_scores_diff = mastery_scores[0] - mastery_scores[1]
        winner = np.array(df["winner"].iloc[match], dtype="int")[np.newaxis]
        dataset[i] = np.concatenate((champions, mastery_scores, winner))

    np.savetxt("pre7.csv", dataset, delimiter=",", fmt="%.5g")


def build_model_pre8(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 275))
    for i, match in enumerate(xrange(0, df.shape[0] - 10, 10)):
        print(i + 1)  # Current processing match

        champions = onehot_champions(df[match:match + 10], db)
        champion_masteries = champion_masteries_team(df[match:match + 10], 
                                                     cursor)
       #champion_masteries_diff = champion_masteries[0] - champion_masteries[1]
        winner = np.array(df["winner"].iloc[match], dtype="int")[np.newaxis]
        dataset[i] = np.concatenate((champions, champion_masteries, winner))

    np.savetxt("pre8.csv", dataset, delimiter=",", fmt="%.5g")


def build_model_pre9(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 283))
    for i, match in enumerate(xrange(0, df.shape[0] - 10, 10)):
        print(i + 1)  # Current processing match

        champions = onehot_champions(df[match:match + 10], db)
        champion_masteries = champion_masteries_summoner(df[match:match + 10], 
                                                         cursor)
        winner = np.array(df["winner"].iloc[match], dtype="int")[np.newaxis]
        dataset[i] = np.concatenate((champions, champion_masteries, winner))

    np.savetxt("pre9.csv", dataset, delimiter=",", fmt="%.5g")


def team_features_zero_to_ten(match, cursor):
    get_features = ("SELECT PL.summonerId, PTD._type, PTD.zeroToTen "
                    "FROM MatchParticipant PA, MatchPlayer PL, "
                    "MatchParticipantTimeline PT, "
                    "MatchParticipantTimelineData PTD "
                    "WHERE PL.summonerId = %s AND PA._match_id = %s "
                    "AND PL._participant_id = PA._id "
                    "AND PA._id = PT._participant_id "
                    "AND PT._id = PTD._timeline_id")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_zero_to_ten = np.zeros(4)
    red_zero_to_ten = np.zeros(4)

    for _, player in blue_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                blue_zero_to_ten[0] += features[2]
            elif features[1] == "damageTakenPerMinDeltas":
                blue_zero_to_ten[1] += features[2]
            elif features[1] == "goldPerMinDeltas":
                blue_zero_to_ten[2] += features[2]
            elif features[1] == "xpPerMinDeltas":
                blue_zero_to_ten[3] += features[2]

    for _, player in red_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                red_zero_to_ten[0] += features[2]
            elif features[1] == "damageTakenPerMinDeltas":
                red_zero_to_ten[1] += features[2]
            elif features[1] == "goldPerMinDeltas":
                red_zero_to_ten[2] += features[2]
            elif features[1] == "xpPerMinDeltas":
                red_zero_to_ten[3] += features[2]

    zero_to_ten = np.concatenate((blue_zero_to_ten, red_zero_to_ten))

    return zero_to_ten



def build_model_pre_in1(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 281))
    for i, match in enumerate(xrange(0, df.shape[0] - 10, 10)):
        print(i + 1)  # Current processing match

        champions = onehot_champions(df[match:match + 10], db)
        zero_to_ten = team_features_zero_to_ten(df[match:match + 10], cursor)
        if zero_to_ten is None:
            continue
        #zero_to_ten_diff = zero_to_ten[:4] - zero_to_ten[4:]
        winner = np.array(df["winner"].iloc[match], dtype="int")[np.newaxis]
        dataset[i] = np.concatenate((champions, zero_to_ten, winner))

    np.savetxt("prein1.csv", dataset, delimiter=",", fmt="%.5g")


def main(args):
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", 
                         db="lol")
    cursor = db.cursor()

    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    feature_models = {"pre1": build_model_pre1,
                      "pre2": build_model_pre2,
                      "pre3": build_model_pre3,
                      "pre5": build_model_pre5,
                      "pre6": build_model_pre6,
                      "pre7": build_model_pre7,
                      "pre8": build_model_pre8,
                      "pre9": build_model_pre9,
                      "prein1": build_model_pre_in1}
    model = feature_models[args[0]](db, cursor)

    cursor.close()
    db.close()


if __name__ == "__main__":
    main(sys.argv[1:])