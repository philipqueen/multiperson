import numpy as np


from multiperson.main import calculate_epipolar_lines


def test_calculate_epipolar_lines_case_1():
    fundamental = np.array(
        [
            [0.00016818481689172314, -0.00255025610913885, 0.4281713138958796],
            [-0.0010647676484824497, 0.0007064693625666324, -2.6929237272934716],
            [0.17155309483547204, 2.1873631440229837, 431.2246998802068],
        ]
    )
    points = np.array(
        [
            [329.0923261642456, 179.47002410888672, 1.0],
            [348.1106472015381, 168.6107635498047, 1.0],
            [353.5486650466919, 170.74283599853516, 1.0],
            [360.5278158187866, 173.56460571289062, 1.0],
        ]
    )

    expected_lines = np.array(
        [
            [
                0.008854363112083005,
                0.019258894671646206,
                0.017697926590265766,
                0.01562759257712692,
            ],
            [
                -0.9999607993585946,
                -0.9998145302885062,
                -0.9998433794322017,
                -0.9998778817186843,
            ],
            [
                301.80040771085413,
                291.9370565249566,
                293.4189741767774,
                295.3832960282752,
            ],
        ]
    )

    calculated_lines = calculate_epipolar_lines(fundamental, points)
    assert np.allclose(calculated_lines, expected_lines)
    assert np.allclose(np.linalg.norm(calculated_lines[:2], axis=0), 1.0)

def test_calculate_epipolar_lines_case_2():
    fundamental = np.array(
        [
            [0.0002244399640885275, -1.1545013533459522e-05, -0.34740444229533385],
            [-0.0009194406858179415, 0.000333753295890008, 1.315160633698539],
            [0.11536707295193288, -1.7237254970204934, 469.16343650977274],
        ]
    )
    points = np.array(
        [
            [258.5925221443176, 306.18343353271484, 1.0],
            [265.3283429145813, 296.1101722717285, 1.0],
            [269.40420627593994, 297.4910545349121, 1.0],
            [274.36509132385254, 299.17016983032227, 1.0],
        ]
    )

    expected_lines = np.array(
        [
            [
                -0.24098918718990694,
                -0.24157077316848066,
                -0.2415076048416421,
                -0.24143032464515318,
            ],
            [
                0.9705278005588238,
                0.970383203456646,
                0.9703989266294833,
                0.9704181564365622,
            ],
            [
                -23.678985350921497,
                -8.82364262278754,
                -10.43724691109805,
                -12.411296455602539,
            ],
        ]
    )

    calculated_lines = calculate_epipolar_lines(fundamental, points)
    assert np.allclose(calculated_lines, expected_lines)
    assert np.allclose(np.linalg.norm(calculated_lines[:2], axis=0), 1.0)
