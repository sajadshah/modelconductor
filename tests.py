import pickle
import unittest
from handlers import SklearnModelHandler, FMUModelHandler, IncomingMeasurementListener
from handlers import Experiment
from handlers import ModelHandler
from matplotlib import pyplot as plt

class SklearnTests(unittest.TestCase):

    def test_iterate_and_plot(self):
        with open('..\\src\\nox_idx.pickle', 'rb') as pickle_file:
            idx = pickle.load(pickle_file)

        with open('..\\src\\data.pickle', 'rb') as pickle_file:
            data = pickle.load(pickle_file)

        X = data[idx].to_numpy()[600:700]
        y = data["Left_NOx"].to_numpy()[600:700]

        sk_hand = SklearnModelHandler(
            model_filename='..\\src\\nox_rfregressor.pickle')

        def step_iterate(X, y):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            plt.ion()
            plt.show()
            for i in range(X.shape[0]):
                xrow = X[i, :]
                yrow = y[i]
                result = list(sk_hand.model.predict(xrow.reshape(1, -1)))
                # print(result[0], yrow)

                # plotting demo
                x1.append(i / 10)
                x2.append(i / 10)
                y1.append(yrow)
                y2.append(result[0])
                ax.clear()
                ax.plot(x1, y1)
                ax.plot(x2, y2)
                plt.draw()
                plt.legend(['NOx, measured [ppm]', 'Nox simulated [ppm]'])
                plt.title('Engine-out NOx vs NRTC Cycle')
                plt.xlabel('Time (s)')
                plt.pause(0.001)

        step_iterate(X, y)
        sk_hand.destroy()


class FmuTests(unittest.TestCase):
    def some_test(self):
        fmu_hand = FMUModelHandler(
            fmu_filename='C:\\Users\\paho\\Dropbox\\Opiskelu\\Diplomityö\\src\\fmi_simulink_demo\\compute_power_5_2.fmu',
            start_time=0.0,
            threshold=2.0,
            stop_time=1238,
            step_size=1
        )

class IncominmgMeasurementTests(unittest.TestCase):
    def some_test(self):
        meas_hand = IncomingMeasurementListener()

class ExperimentTests(unittest.TestCase):

    def test_single_source_and_consumer_can_be_added(self):
        listen = IncomingMeasurementListener()
        mdl = ModelHandler()
        ex = Experiment()

        ex.add_route((listen, mdl))

        self.assertEqual(len(ex.routes), 1)
        self.assertListEqual(ex.routes, [(listen, mdl)])

    def test_single_source_and_consumer_setup(self):
        listen = IncomingMeasurementListener()
        mdl = ModelHandler()
        ex = Experiment()
        ex.add_route((listen, mdl))
        ex.setup()

        self.assertIs(mdl.sources[0], listen)
        self.assertIs(listen.consumers[0], mdl)




if __name__ == '__main__':
    unittest.main()