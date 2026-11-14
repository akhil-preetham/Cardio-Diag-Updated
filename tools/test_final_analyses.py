import numpy as np
import pytest
from finalize_analyses import threshold_for_sensitivity, net_benefit, subset


def test_threshold_attains_requested_sensitivity_in_training_only():
    y = np.array([1]*10+[0]*10)
    p = np.linspace(.01,.99,20)
    t = threshold_for_sensitivity(y,p,.9)
    assert np.mean(p[y==1]>=t) >= .9
    assert np.mean(p[y==1]>t) < .9


def test_net_benefit_matches_hand_count():
    assert net_benefit([1,1,0,0],[.9,.6,.7,.1],.5) == pytest.approx(.25)
    assert net_benefit([1,1,0,0],[0,0,0,0],.5) == 0
    with pytest.raises(ValueError): net_benefit([0,1],[.1,.9],1)


def test_learning_subset_cannot_include_evaluation_records():
    y = np.tile([0,1],50)
    train = np.arange(80)
    for f in [.1,.25,.5,.75,1]:
        selected = subset(train,y,f,17)
        assert set(selected) <= set(train)
        assert not set(selected) & set(range(80,100))
