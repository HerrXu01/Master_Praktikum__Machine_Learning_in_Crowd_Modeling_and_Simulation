import json


class Scenariofile():

    def read_scenariofile(filename):
        """
        Reading scenario file into variable
        """
        with open(filename,'r') as f:
            add_pedestrian = json.load(f)
        return add_pedestrian

    def write_scenariofile(filename,data):
        """
        Writing the modified data into scenario file
        """
        with open(filename, "w") as json_file:
            json.dump(data, json_file)

    def format_scenariofile(filename):
        """
        Formatting scenario file to input it in Vadere console
        """
        with open(filename,'r') as file:
            filedata = file.read()

        filedata = filedata.replace('"null"','null')
        filedata = filedata.replace('"false"','false')

        with open(filename,'w') as file:
            file.write(filedata)

    def fix_attributes(target_id):
        """
        Defining the other attributes for the pedestrian
        """
        
        dict = { "source" : 'null', 
            "targetIds" : [ target_id ],
            "nextTargetListIndex" : 0,
            "isCurrentTargetAnAgent" : 'false',
            "position" : { "x" : 10.7, "y" : 2.1 },                     # pPositioning edestrian at the required coordinate, user can change it to place it somewhere else
            "velocity" : {
            "x" : 11.0,
            "y" : 1.7
            },
            "freeFlowSpeed" : 1.310473292392256,
            "followers" : [ ],
            "idAsTarget" : -1,
            "isChild" : 'false',
            "isLikelyInjured" : 'false',
            "psychologyStatus" : {
            "mostImportantStimulus" : 'null',
            "threatMemory" : {
                "allThreats" : [ ],
                "latestThreatUnhandled" : 'false'
            },

            "selfCategory" : "TARGET_ORIENTED",
            "groupMembership" : "OUT_GROUP",

            "knowledgeBase" : {
                "knowledge" : [ ],
                "informationState" : "NO_INFORMATION"
            },
            "perceivedStimuli" : [ ],
            "nextPerceivedStimuli" : [ ]
            },

            "healthStatus" : 'null',
            "infectionStatus" : 'null',
            "groupIds" : [ ],
            "groupSizes" : [ ],
            "agentsInGroup" : [ ],

            "trajectory" : {
            "footSteps" : [ ]
            },
            "modelPedestrianMap" : { },
            "type" : "PEDESTRIAN"
        }
        return dict