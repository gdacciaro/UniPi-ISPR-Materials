import graphviz # only for graph visualization
import numpy as np # used for picking random numbers

class BayesianNetwork(object):
    def __init__(self):
        self.nodes = set()

    def add_states(self, list_of_states):
        for node in list_of_states:
            self.nodes.add(node)

    def print(self):
        for node in self.nodes:
            node.print()
            print("")

    def add_probabilities(self, node, input_probabilities):
        # [stage 1] check sum to one
        for p in input_probabilities:
            sum_at_one = 0
            for single_prob in p.probabilities:
                sum_at_one+= single_prob
            if sum_at_one != 1:
                raise Exception("Probabilities of " + str(node) + " must sum to 1. [Actual sum = " + str(sum_at_one) + " ]")

        # [stage 2] check causes' domains
        for p in input_probabilities:
            for single_causes in p.causes:
                domain = single_causes[0].domain_values
                actual_value = single_causes[1]
                if actual_value not in domain:
                    raise Exception("The cause "+str(single_causes[0])+" has a not valid value [the value <"+str(single_causes[1])+"> is not in its domain]")

        # [stage 3] check if every domain values have a probability
        for p in input_probabilities:
            if len(p.probabilities)!= len(node.domain_values):
                raise Exception("Not all values in the node "+str(node)+" domain have a probability.")

        # [stage 4] check duplicates in the CPT
        check_duplicate_CPT = []
        for p in input_probabilities:
            if p.causes in check_duplicate_CPT:
                raise Exception("The cause "+str(p.causes)+" of the node "+str(node)+" was found more than once.")
            else:
                check_duplicate_CPT.append(p.causes)

        # [stage 5] check missing item in the CPT
        number_of_items = 1
        for parent in node.parents:
            number_of_items *= len(parent.domain_values)
        if len(input_probabilities) != number_of_items:
            raise Exception("The number of item in the CPT of the node "+str(node)+" is not valid")

        # [stage 6] check validity of causes
        for p in input_probabilities:
            for cause in p.causes:
                if not cause[0] in node.parents:
                    raise Exception("The cause "+str(cause[0])+ " cannot be added to the node "+str(node))

        # if everything's fine, I can add my probabilities to the node
        node.set_prob(input_probabilities)

    def add_edge(self,A,B):
        """ Preconditions """
        if A not in self.nodes:
            raise Exception("The node <" + str(A) + "> is not added to the Bayesian Network")
        if B not in self.nodes:
            raise Exception("The node <" + str(B) + "> is not added to the Bayesian Network")

        A.add_edge(B)

        """ A BN is a DAG, so I raise an exception if I found a cycle"""
        self.check_cycles()

    def __str__(self):
        return self.print()
    def __repr__(self):
        return self.print()

    def __check_cycles_dfs_visit(self, node, visited, recStack):
        visited[node] = True
        recStack[node] = True

        for neighbour in node.children:
            if not visited[neighbour]:
                """ If I visit a node twice, there is a cycle """
                if self.__check_cycles_dfs_visit(neighbour, visited, recStack):
                    return True
            elif recStack[neighbour]:
                """ or it's in my recursion stack, there is a cycle """
                return True

        recStack[node] = False
        return False

    def check_cycles(self):
        """ This function takes care of searching for cycles in the graph.
            If it finds one, it throws an exception because a BN cannot contain cycles.
         """
        result = False

        """ Mark all the vertices as not visited """
        visited = {}
        recStack = {}

        for node in self.nodes:
            visited[node] = False
            recStack[node] = False

        for node in self.nodes:
            if not visited[node]:
                if self.__check_cycles_dfs_visit(node, visited, recStack):
                    result = True

        if result:
            raise Exception("A cycle was found in the Bayesian Network")

    def query(self, param):
        """ This function performs a BFS on the Bayesian network to find the probabilities to be multiplied.
            It starts by visiting independent nodes, and then adds their children to the queue.
            As it visits the queue, it adds the probability given the parents (which have already been visited) to the result.
         """

        """ Preconditions """
        if len(param) != len(self.nodes):
            raise Exception("A query needs every and only the variables in the BN")

        """ The graph visit needs to start from independent nodes """
        independent_nodes = self.get_independent_nodes()

        """ Once we found independent nodes, we put them to the BFS queue """
        queue = independent_nodes

        """ Mark all the vertices as not visited """
        visited = {}
        for node in self.nodes:
            visited[node] = False

        prob_found = list()

        while queue:
            item = queue.pop(0) #Remove from head

            """ I execute this for cycle because I don't want to put constraints in the order of appearance of the random variables in the query """
            input_value = None
            for p in param:
                input_item = p[0]
                value = p[1]
                if item == input_item:
                    input_value = value
                    break

            value_index = item.domain_values.index(input_value)

            """ safety first """
            if not hasattr(item, "prob"):
                raise Exception("No CPT added to the node <"+str(item)+">")

            for p in item.prob:
                selected_probability = p.probabilities[value_index]
                if not p.causes:
                    """If the node is independent"""
                    prob_found.append(selected_probability)
                else:
                    """If the node is dependent, I choose the probability whose causes are present in the causes passed as input to the algorithm."""
                    match_counter = 0
                    for single_cause in p.causes:
                        if  single_cause in param:
                            match_counter+=1

                    if match_counter == len(p.causes):
                        prob_found.append(selected_probability)
                        break

            """ Adding my unvisited children to the queue """
            for child in item.children:
                if not visited[child]:
                    queue.append(child)
                    visited[child] = True

        result = 1
        for prob in prob_found:
            result *= prob

        return result

    def plot(self, filename = "../bayesian_network.dot"):
        g = graphviz.Digraph('G', filename=filename)
        for node in self.nodes:
            for child in node.children:
                g.edge(str(node), str(child))
        try:
            g.view()
        except:
            pass

        print("Plot file (",filename,") saved")

    def get_independent_nodes(self):
        independent_nodes = list()
        for node in self.nodes:
            if node.is_independent():
                independent_nodes.append(node)
        return  independent_nodes

    def ancestral_sampling(self):
        """ The graph visit needs to start from independent nodes """
        independent_nodes = self.get_independent_nodes()

        """ Once we found independent nodes, we put them to the BFS queue """
        queue = independent_nodes

        """ Mark all the vertices as not visited """
        visited = {}
        for node in self.nodes:
            visited[node] = False

        sample_causes = []
        while queue:
            item = queue.pop(0)  # Remove from head
            sample = item.sample(sample_causes)

            sample_causes.append([item, sample])

            """ Adding my unvisited children to the queue """
            for child in item.children:
                if not visited[child]:
                    queue.append(child)
                    visited[child] = True

        return self.query(sample_causes), sample_causes


class BayesianNode(object):

    def __init__(self, tag, domain_values):
        self.children = set()
        self.parents = set()
        self.tag = tag
        self.domain_values = domain_values
        self.prob = None

    def is_independent(self):
        return len(self.parents)==0

    def add_edge(self, child):
        self.children.add(child)
        child.parents.add(self)

    def set_prob(self, prob):
        self.prob = prob

    def well_print_probs(self):
        if self.prob is not None:
            for p in self.prob:
                if p.causes:
                    print(p.causes,"\t\t", p.probabilities)
                else:
                    print("NO CAUSES", "\t\t", p.probabilities)

    def print(self):
        print("TAG: ", self.tag , " | DOMAIN: ", self.domain_values)
        print("CHILDEN: ", self.children)
        print("CPT:")
        self.well_print_probs()

    def sample(self, selected_causes = None):
        #Precondition for dependent nodes:
        for p in self.prob:
            if p.causes and selected_causes is None:
                raise Exception("You can't sample the node "+str(self)+" without pass the causes")

        selected_prob = None

        for p in self.prob:
            if p.causes: # If the causes list is not empty

                #The selected_causes input list may contain more causes than are needed for this node
                    #So, I check if all my causes are conteined into the input list, ignoring all the others

                all_causes_are_selected = 0
                for cause in p.causes: # For each my cause ..

                    for single_selected_cause in selected_causes:
                        if cause == single_selected_cause: # .. checks if it's contained
                            all_causes_are_selected+=1

                #If all the causes are contained, I found my probabilities
                if len(p.causes) == all_causes_are_selected:
                    selected_prob = p.probabilities
                    break
            else:
                """ independent nodes """
                selected_prob = p.probabilities
                break

        picked_value = np.random.choice(self.domain_values, 1, p=selected_prob)
        return picked_value[0]

    def __str__(self):
        return self.tag
    def __repr__(self):
        return self.tag

class CPT_Item(object):
    def __init__(self, causes, probabilities):
        self.causes = causes
        self.probabilities = probabilities
