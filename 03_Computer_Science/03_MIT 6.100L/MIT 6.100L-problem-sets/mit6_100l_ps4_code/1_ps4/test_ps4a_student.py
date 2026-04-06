import unittest
from tree import Node
import ps4a as student

# A class that inherits from unittest.TestCase, where each function
# is a test you want to run on the student's code. For a full description
# plus a list of all the possible assert methods you can use, see the
# documentation: https://docs.python.org/3/library/unittest.html#unittest.TestCase
class TestPS4A(unittest.TestCase):
    ### test representation
    def test_data_representation(self):
        str_representation = ['      8      \n\n    /   \\    \n\n  2       10  \n\n/   \\        \n\n1   6        ',
                                '              7              \n\n          /       \\          \n\n      2               9      \n\n    /   \\           /   \\    \n\n  1       5       8       10  \n\n        /   \\                \n\n        3   6                ',
                                '              5              \n\n          /       \\          \n\n      3               14      \n\n    /   \\           /   \\    \n\n  2       4       12       21  \n\n                        /   \\\n\n                        20   26']
        student_t1 = student.tree1
        student_t2 = student.tree2
        student_t3 = student.tree3

        self.assertTrue(str(student_t1) == str_representation[0], "\nYour Tree: \n{}\nCorrect Tree\n{}".format(student_t1, str_representation[0]))
        self.assertTrue(str(student_t2) == str_representation[1], "\nYour Tree: \n{}\nCorrect Tree\n{}".format(student_t2, str_representation[1]))
        self.assertTrue(str(student_t3) == str_representation[2], "\nYour Tree: \n{}\nCorrect Tree\n{}".format(student_t3, str_representation[2]))

    def test_tree_height(self):
        answers = [5, 6, 2, 0]
        tr1 = Node(13, Node(10, Node(5, Node(4), Node(6)), Node(11)), Node(15, right_child=Node(16,Node(3,Node(20,Node(17))))))
        tr2 = Node(37, Node(24, Node(7, Node(2, right_child = Node(5))), Node(32)), Node(42, Node(40), Node(42, right_child=Node(43,Node(2,Node(14,Node(30)))))))
        tr3 = Node(5, Node(1), Node(5, Node(5)))

        st_tr1 = student.find_tree_height(tr1)
        st_tr2 = student.find_tree_height(tr2)
        st_tr3 = student.find_tree_height(tr3)

        self.assertEqual(st_tr1, answers[0], "Expected {} but got {}".format(answers[0], st_tr1))
        self.assertEqual(st_tr2, answers[1], "Expected {} but got {}".format(answers[1], st_tr2))
        self.assertEqual(st_tr3, answers[2], "Expected {} but got {}".format(answers[2], st_tr3))

    def test_tree_height_additional(self):
        answers = [4, 6, 0]
        tr1 = Node(5, Node(7, Node(9, Node(10, Node(11)))))
        tr2 = Node(5, right_child=Node(7, right_child=Node(9, right_child=Node(10, right_child=Node(11, right_child=Node(90,Node(2)))))))
        tr3 = Node(5)

        st_tr1 = student.find_tree_height(tr1)
        st_tr2 = student.find_tree_height(tr2)
        st_tr3 = student.find_tree_height(tr3)

        self.assertEqual(st_tr1, answers[0], "Expected {} but got {}".format(answers[0], st_tr1))
        self.assertEqual(st_tr2, answers[1], "Expected {} but got {}".format(answers[1], st_tr2))
        self.assertEqual(st_tr3, answers[2], "Expected {} but got {}".format(answers[2], st_tr3))

    def test_is_min_heap(self):
        tr1 = Node(5,Node(15,None,Node(16,Node(30),Node(17))),Node(6,Node(20,None,Node(45)),Node(11)))
        tr2 = Node(2,Node(3,Node(4),Node(5,Node(6))),Node(7,None,Node(8,Node(9),Node(1))))
        compare_func = lambda x,y: x > y
        st_tr1 = student.is_heap(tr1,compare_func)
        self.assertTrue(st_tr1,"Expected True but got False")
        st_tr2 = student.is_heap(tr2,compare_func)
        self.assertFalse(st_tr2,"Expected False but got True")

    def test_is_max_heap(self):
        tr1 = Node(15,Node(4,Node(3,None,Node(2)),Node(1)),Node(11,Node(10),Node(7,Node(5))))
        tr2 = Node(10,Node(7,None,Node(4,Node(3,None,Node(5)))))
        compare_func = lambda x,y: x < y
        st_tr1 = student.is_heap(tr1,compare_func)
        self.assertTrue(st_tr1,"Expected True but got False")
        st_tr2 = student.is_heap(tr2,compare_func)
        self.assertFalse(st_tr2,"Expected False but got True")


# Dictionary mapping function names from the above TestCase class to
# the point value each test is worth.
point_values = {
	'test_data_representation' : 20,
	'test_tree_height' : 20,
	'test_tree_height_additional' : 20,
    'test_is_max_heap' : 20,
    'test_is_min_heap': 20
}

# Subclass to track a point score and appropriate
# grade comment for a suit of unit tests
class Results_600(unittest.TextTestResult):

    # We override the init method so that the Result object
    # can store the score and appropriate test output.
    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = 100

    def addFailure(self, test, err):
        test_name = test._testMethodName
        msg = str(err[1])
        self.handleDeduction(test_name, msg)
        super(Results_600, self).addFailure(test, err)

    def addError(self, test, err):
        test_name = test._testMethodName
        self.handleDeduction(test_name, None)
        super(Results_600, self).addError(test, err)

    def handleDeduction(self, test_name, message):
        point_value = point_values[test_name]
        if message is None:
            message = 'Your code produced an error on test %s.' % test_name
        self.output.append('[-%s]: %s' % (point_value, message))
        self.points -= point_value

    def getOutput(self):
        if len(self.output) == 0:
            return "All correct!"
        return '\n'.join(self.output)

    def getPoints(self):
        return self.points

if __name__ == '__main__':

	suite = unittest.TestSuite()
	suite.addTest(unittest.makeSuite(TestPS4A))
	result = unittest.TextTestRunner(verbosity=2, resultclass=Results_600).run(suite)

	output = result.getOutput()
	points = result.getPoints()

	# weird bug with rounding
	if points < .1:
		points = 0

	print("\nProblem Set 4A Unit Test Results:")
	print(output)
	print("\n{}% done with part A".format(points))
