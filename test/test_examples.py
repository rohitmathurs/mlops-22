def test_equal():
	assert 1 == 1
	
def test_avg():
	a = 5
	b = 6
	c = 7
	len = 3
	avg = (a + b + c) / len
	expected_avg = 6
	assert avg == expected_avg
