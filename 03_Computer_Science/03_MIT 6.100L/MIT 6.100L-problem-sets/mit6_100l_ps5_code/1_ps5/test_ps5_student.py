import sys
import unittest
import ps5
import pickle
from PIL import ImageChops

class TestPS5(unittest.TestCase):
    # this is a helper to save hidden images for the tester to use to validate
    # student code that creates the hidden image.
    # NOTE: students should not run this function as it is not necessary
    # for the pset and can corrupt the tester from working properly. 
    # This should only be used by staff to update the tester if/when the 
    # hidden images change.
    def help_save_hidden_img(self, pkl_file_name, img_file_name):
        f = open(pkl_file_name, 'wb')
        im = reveal_image(img_file_name)
        pickle.dump(im, f)
        f.close()
        
    def are_imgs_equal(self, im1, im2):
        return ImageChops.difference(im1, im2).getbbox() is None
    
    def help_check_hidden_img(self, pkl_file_name, student_img):
        f = open(pkl_file_name, 'rb')
        im = pickle.load(f)
        f.close()
        return self.are_imgs_equal(im, student_img)
        
    def bit_tester(self, f, test_cases):
        for inputs, expected in test_cases:
            actual = f(*inputs)
            err_msg = "Method {methodname} produced an incorrect result with inputs {inputs}. Got {actual} but expected {expected}.".format(
                methodname=f.__name__,
                inputs=inputs,
                actual=actual,
                expected=expected
            )
            self.assertEqual(actual, expected, err_msg)


    def test_extract_end_bits(self):
        test_cases = [
            ((2, 3), 3),
            ((1, 0), 0),
            ((5, 214), 22),
            ((4, 135), 7),
            ((4, 235), 11),
            ((3, (214,17,8)), (6,1,0)),
            ((4, (135,88,17)), (7,8,1)),
            ((6, (255,255,255)), (63,63,63))
        ]
        self.bit_tester(ps5.extract_end_bits, test_cases)
    
    def test_reveal_img_bw(self):
        self.assertTrue(self.help_check_hidden_img('tester_bw_img.obj', 
                                                   ps5.reveal_image('hidden1.bmp')),
                        "Hidden black/white image not revealed correctly")
        
    def test_reveal_img_rgb(self):
        self.assertTrue(self.help_check_hidden_img('tester_rgb_img.obj',
                                                   ps5.reveal_image('hidden2.bmp')),
                        "Hidden rgb (color) image not revealed correctly")

point_values = {
    'test_extract_end_bits' : 1,
    'test_reveal_img_bw' : 1,
    'test_reveal_img_rgb' : 1
}


class Results_600(unittest.TextTestResult):
    # We override the init method so that the Result object
    # can store the score and appropriate test output.
    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = sum(point_values.values())
        self.total_points = sum(point_values.values())

    def addFailure(self, test, err):
        test_name = test._testMethodName
        msg = str(err[1])
        self.handleDeduction(test_name, msg)
        super(Results_600, self).addFailure(test, err)

    def addError(self, test, err):
        test_name = test._testMethodName
        msg = 'Your code produced an error on test %s: %s' % (test_name, str(err[1]))
        self.handleDeduction(test_name, msg)
        super(Results_600, self).addError(test, err)

    def handleDeduction(self, test_name, message):
        point_value = point_values[test_name]
        self.output.append('[-%s]: %s' % (point_value, message))
        self.points -= point_value

    def getOutput(self):
        if len(self.output) == 0:
            return "All correct!"
        return '\n'.join(self.output)

    def getPoints(self):
        return round(self.points, 2)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPS5))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Results_600).run(suite)

    output = result.getOutput()
    points = result.getPoints()

    print("\n\nProblem Set 5 Unit Test Results:")
    print(output)
    print("Points: {points}/{total_points}".format(points=points, total_points=result.total_points))
