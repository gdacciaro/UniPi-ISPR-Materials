from bayesian_network import BayesianNode, BayesianNetwork, CPT_Item

if __name__ == '__main__':
    burglary    = BayesianNode("burglary", domain_values=[True, False])
    earthquake  = BayesianNode("earthquake", domain_values=[True, False])
    alarm = BayesianNode("alarm", domain_values=[True, False])
    david_calls  = BayesianNode("david_calls", domain_values=[True, False])
    sophia_calls = BayesianNode("sophia_calls", domain_values=[True, False])

    bn = BayesianNetwork()

    bn.add_states([burglary, earthquake, alarm, david_calls, sophia_calls])

    bn.add_edge(burglary, alarm)
    bn.add_edge(earthquake, alarm)
    bn.add_edge(alarm, david_calls)
    bn.add_edge(alarm, sophia_calls)

    bn.add_probabilities(burglary,
            {CPT_Item(causes=[], probabilities=[0.001, 0.999])}
    )

    bn.add_probabilities(earthquake,
            {CPT_Item(causes=[], probabilities=[0.002, 0.998])}
    )


    bn.add_probabilities(alarm,
            {
                CPT_Item(causes=[[burglary, True], [earthquake, True]], probabilities=[0.95, 0.05]),
                CPT_Item(causes=[[burglary, True], [earthquake, False]], probabilities=[0.94, 0.06]),

                CPT_Item(causes=[[burglary, False], [earthquake, True]], probabilities=[0.29, 0.71]),
                CPT_Item(causes=[[burglary, False], [earthquake, False]], probabilities=[0.001, 0.999]),
           }
    )

    bn.add_probabilities(david_calls,
          {
              CPT_Item(causes=[[alarm, True]], probabilities=[0.90, 0.1]),
              CPT_Item(causes=[[alarm, False]], probabilities=[0.05, 0.95]),
          }
    )

    bn.add_probabilities(sophia_calls,
         {
             CPT_Item(causes=[[alarm, True]], probabilities=[0.7, 0.3]),
             CPT_Item(causes=[[alarm, False]], probabilities=[0.01, 0.99]),
         }
    )

    bn.plot()
    print(bn.query([[david_calls, True], [sophia_calls, True], [alarm, False], [earthquake, False], [burglary, False]]))
