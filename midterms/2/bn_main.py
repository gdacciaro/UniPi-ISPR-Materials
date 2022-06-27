#Implement a Bayesian Network (BN) comprising at least 10 nodes, all with binomial or multinomial distribution.
#Represent the BN with the data structures that you deem appropriate and in the programming language that you prefer.
#The BN should model some problem/process of your choice, so you are also free to define the topology
#   according to your prior knowledge (just be ready to justify your choices).
# For instance, you can define a BN to represent a COVID diagnosis through a certain number of events/exams/symptoms:
#   e.g. Cough, Cold, Fever, Breathing problems, Swab Exam, etc.
# Or you can model your daily routine: Wakeup, Coffee, Toilet, Study, Lunch, etc.
#Once you have modelled the BN, also plug in the necessary local conditional probability tables.
#You can set the values of the probabilities following your own intuition on the problem (ie no need to learn them from data).
#Then run some episoded of Ancestral Sampling on the BN and discuss the results.

#The assignment needs to be fully implemented by you, without using BN libraries.

from bayesian_network import BayesianNode, BayesianNetwork, CPT_Item

import matplotlib.pyplot as plt

if __name__ == '__main__':

    studying_m1 = BayesianNode("studying_m1", domain_values=["Hard", "Sufficient", "Not so much"])
    studying_m2 = BayesianNode("studying_m2", domain_values=["Hard", "Sufficient", "Not so much"])
    studying_m3 = BayesianNode("studying_m3", domain_values=["Hard", "Sufficient", "Not so much"])
    studying_m4 = BayesianNode("studying_m4", domain_values=["Hard", "Sufficient", "Not so much"])
    studying_oral = BayesianNode("studying_oral", domain_values=["Hard", "Sufficient", "Not so much"])

    midterm1 = BayesianNode("midterm1", domain_values=[True, False])
    midterm2 = BayesianNode("midterm2", domain_values=[True, False])
    midterm3 = BayesianNode("midterm3", domain_values=[True, False])
    midterm4 = BayesianNode("midterm4", domain_values=[True, False])

    oral = BayesianNode("oral", domain_values=["Honorary", "Remarkable", "Pass", "Fail"])

    happy_gennaro_daniele = BayesianNode("happy_gennaro_daniele", domain_values=[True, False])

    bn = BayesianNetwork()

    bn.add_states([studying_m1, studying_m2, studying_m3, studying_m4, studying_oral,
                   midterm1, midterm2, midterm3, midterm4,
                   oral, happy_gennaro_daniele])

    bn.add_edge(studying_m1, studying_m2)
    bn.add_edge(studying_m2, studying_m3)
    bn.add_edge(studying_m3, studying_m4)
    bn.add_edge(studying_m4, studying_oral)

    bn.add_edge(studying_m1, midterm1)
    bn.add_edge(studying_m2, midterm2)
    bn.add_edge(studying_m3, midterm3)
    bn.add_edge(studying_m4, midterm4)
    bn.add_edge(studying_oral, oral)

    bn.add_edge(midterm1, midterm2)
    bn.add_edge(midterm2, midterm3)
    bn.add_edge(midterm3, midterm4)
    bn.add_edge(midterm4, oral)

    bn.add_edge(oral, happy_gennaro_daniele)

    bn.add_probabilities(studying_m1,
         {CPT_Item(causes=[], probabilities=[0.8, 0.17, 0.03])}
    )

    bn.add_probabilities(studying_m2,
         {
             CPT_Item(causes=[[studying_m1, "Hard"]], probabilities=[0.9, 0.07, 0.03]),
             CPT_Item(causes=[[studying_m1, "Sufficient"]], probabilities=[0.5, 0.4, 0.1]),
             CPT_Item(causes=[[studying_m1, "Not so much"]], probabilities=[0.05, 0.3, 0.65]),
         }
    )

    bn.add_probabilities(studying_m3,
        {
            CPT_Item(causes=[[studying_m2, "Hard"]], probabilities=[0.9, 0.07, 0.03]),
            CPT_Item(causes=[[studying_m2, "Sufficient"]], probabilities=[0.5, 0.4, 0.1]),
            CPT_Item(causes=[[studying_m2, "Not so much"]], probabilities=[0.05, 0.3, 0.65]),
        }
    )

    bn.add_probabilities(studying_m4,
        {
             CPT_Item(causes=[[studying_m3, "Hard"]], probabilities=[0.9, 0.07, 0.03]),
             CPT_Item(causes=[[studying_m3, "Sufficient"]], probabilities=[0.5, 0.4, 0.1]),
             CPT_Item(causes=[[studying_m3, "Not so much"]], probabilities=[0.05, 0.3, 0.65]),
        }
    )

    bn.add_probabilities(studying_oral,
        {
            CPT_Item(causes=[[studying_m4, "Hard"]], probabilities=[0.9, 0.07, 0.03]),
            CPT_Item(causes=[[studying_m4, "Sufficient"]], probabilities=[0.5, 0.4, 0.1]),
            CPT_Item(causes=[[studying_m4, "Not so much"]], probabilities=[0.05, 0.3, 0.65]),
        }
    )

    bn.add_probabilities(midterm1,
          {
              CPT_Item(causes=[[studying_m1, "Hard"]], probabilities=[0.9, 0.1]),
              CPT_Item(causes=[[studying_m1, "Sufficient"]], probabilities=[0.75, 0.25]),
              CPT_Item(causes=[[studying_m1, "Not so much"]], probabilities=[0.05, 0.95]),
          }
    )

    bn.add_probabilities(midterm2,
         {
             CPT_Item(causes=[[studying_m2, "Hard"], [midterm1, True]], probabilities=[0.99, 0.01]),
             CPT_Item(causes=[[studying_m2, "Sufficient"], [midterm1, True]], probabilities=[0.75, 0.25]),
             CPT_Item(causes=[[studying_m2, "Not so much"], [midterm1, True]], probabilities=[0.06, 0.94]),

             CPT_Item(causes=[[studying_m2, "Hard"], [midterm1, False]], probabilities=[0, 1]),
             CPT_Item(causes=[[studying_m2, "Sufficient"], [midterm1, False]], probabilities=[0, 1]),
             CPT_Item(causes=[[studying_m2, "Not so much"], [midterm1, False]], probabilities=[0, 1]),
         }
    )



    bn.add_probabilities(midterm3,
         {
             CPT_Item(causes=[[studying_m3, "Hard"], [midterm2, True]], probabilities=[0.9, 0.1]),
             CPT_Item(causes=[[studying_m3, "Sufficient"], [midterm2, True]], probabilities=[0.75, 0.25]),
             CPT_Item(causes=[[studying_m3, "Not so much"], [midterm2, True]], probabilities=[0.05, 0.95]),

             CPT_Item(causes=[[studying_m3, "Hard"], [midterm2, False]], probabilities=[0, 1]),
             CPT_Item(causes=[[studying_m3, "Sufficient"], [midterm2, False]], probabilities=[0, 1]),
             CPT_Item(causes=[[studying_m3, "Not so much"], [midterm2, False]], probabilities=[0, 1]),
         }
    )

    bn.add_probabilities(midterm4,
        {
            CPT_Item(causes=[[studying_m4, "Hard"], [midterm3, True]], probabilities=[0.8, 0.2]),
            CPT_Item(causes=[[studying_m4, "Sufficient"], [midterm3, True]], probabilities=[0.65, 0.35]),
            CPT_Item(causes=[[studying_m4, "Not so much"], [midterm3, True]], probabilities=[0.04, 0.96]),

            CPT_Item(causes=[[studying_m4, "Hard"], [midterm3, False]], probabilities=[0, 1]),
            CPT_Item(causes=[[studying_m4, "Sufficient"], [midterm3, False]], probabilities=[0, 1]),
            CPT_Item(causes=[[studying_m4, "Not so much"], [midterm3, False]], probabilities=[0, 1]),
        }
    )

    bn.add_probabilities(oral,
        {
             CPT_Item(causes=[[studying_oral, "Hard"], [midterm4, True]], probabilities=[0.6, 0.25, 0.1, 0.05]),
             CPT_Item(causes=[[studying_oral, "Sufficient"], [midterm4, True]], probabilities=[0.2, 0.35, 0.4, 0.05]),
             CPT_Item(causes=[[studying_oral, "Not so much"], [midterm4, True]], probabilities=[0., 0.1, 0.3, 0.6]),

             CPT_Item(causes=[[studying_oral, "Hard"], [midterm4, False]], probabilities=[0, 0, 0, 1]),
             CPT_Item(causes=[[studying_oral, "Sufficient"], [midterm4, False]], probabilities=[0, 0, 0, 1]),
             CPT_Item(causes=[[studying_oral, "Not so much"], [midterm4, False]], probabilities=[0, 0, 0, 1]),
        }
    )

    bn.add_probabilities(happy_gennaro_daniele,
          {
              CPT_Item(causes=[[oral, "Honorary"]], probabilities=[1, 0.]),
              CPT_Item(causes=[[oral, "Remarkable"]], probabilities=[1, 0.]),
              CPT_Item(causes=[[oral, "Pass"]], probabilities=[0.6, 0.4]),
              CPT_Item(causes=[[oral, "Fail"]], probabilities=[0.3, 0.7]),
          }
    )

    bn.plot()

    import time
    number_of_sample = 100
    #for number_of_sample in [3,5,10,20,30,50,100,150,300,500,1000,2000,5000]:
    ts = time.time()
    fail_count = 0
    pass_count = 0
    remarkable_count = 0
    honorary_count = 0

    for i in range(number_of_sample):
        sample = bn.ancestral_sampling()
        oral = sample[1][-2]
        prob = sample[0]
        if oral[1] == "Fail":
            fail_count+=1
        elif oral[1] == "Pass":
            pass_count+=1
        elif oral[1] == "Remarkable":
            remarkable_count+=1
        else:
            honorary_count+=1

    exec_time = str(round(time.time()-ts, 3))+" sec"

    data = {'Fail': fail_count, 'Pass': pass_count,
            'Remarkable': remarkable_count, 'Honorary': honorary_count}

    labels = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter(number_of_sample))
    plt.xticks(size=25)
    plt.bar(labels, values, color='maroon', width=0.3)
    plt.suptitle("Number of samples: "+str(number_of_sample)+" | "+str(exec_time), size=25)
    plt.grid(color='gray', linestyle='dashed')
    #plt.savefig(str(number_of_sample)+'.png')
    plt.show()

    print(data)
