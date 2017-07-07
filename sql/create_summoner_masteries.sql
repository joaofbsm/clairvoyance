USE lol;

CREATE TABLE SummonerChampMasteries (summId INT(11), championId INT(11), mastery INT(11), PRIMARY KEY(summId, championId), FOREIGN KEY(summId) REFERENCES Summoner(id) ON DELETE CASCADE, FOREIGN KEY(championId) REFERENCES Champion(id) ON DELETE CASCADE);
CREATE TABLE SummonerMasteries (summId INT(11), mastery INT(11), PRIMARY KEY(summId), FOREIGN KEY(summId) REFERENCES Summoner(id) ON DELETE CASCADE);