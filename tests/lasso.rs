use lasso::{*};
use lasso::linalg::{Matrix};

#[test]
fn simple_regression_start_at_solution() {
	let mut test_data: Matrix<f64> = Matrix::new(3, 3);
	test_data.set(1, 0, 1.0).unwrap();
	test_data.set(2, 0, 2.0).unwrap();
	test_data.set(0, 1, 3.0).unwrap();
	test_data.set(1, 1, 4.0).unwrap();
	test_data.set(2, 1, 5.0).unwrap();
	test_data.set(0, 2, 6.0).unwrap();
	test_data.set(1, 2, 7.0).unwrap();
	test_data.set(2, 2, 8.0).unwrap();

	let mut test_result: Matrix<f64> = Matrix::new(1, 3);
	test_result.set(0, 0, 5.0).unwrap();
	test_result.set(0, 1, 14.0).unwrap();
	test_result.set(0, 2, 23.0).unwrap();

	let mut initial_w: Matrix<f64> = Matrix::new(1, 3);
	initial_w.set(0, 0, 0.0).unwrap();
	initial_w.set(0, 1, 1.0).unwrap();
	initial_w.set(0, 2, 2.0).unwrap();

	let result = simple_lasso(&test_data, &test_result, &initial_w, 0.5, 0.0, 1000).unwrap();
	assert_eq!(result.get(0, 0).unwrap(), 0.0);
	assert_eq!(result.get(0, 1).unwrap(), 1.0);
	assert_eq!(result.get(0, 2).unwrap(), 2.0);
}

#[test]
fn simple_regression_start_not_solution() {
	let mut test_data: Matrix<f64> = Matrix::new(3, 3);
	test_data.set(1, 0, 1.0).unwrap();
	test_data.set(2, 0, 2.0).unwrap();
	test_data.set(0, 1, 3.0).unwrap();
	test_data.set(1, 1, 4.0).unwrap();
	test_data.set(2, 1, 5.0).unwrap();
	test_data.set(0, 2, 6.0).unwrap();
	test_data.set(1, 2, 7.0).unwrap();
	test_data.set(2, 2, 8.0).unwrap();

	let mut test_result: Matrix<f64> = Matrix::new(1, 3);
	test_result.set(0, 0, 5.0).unwrap();
	test_result.set(0, 1, 14.0).unwrap();
	test_result.set(0, 2, 23.0).unwrap();

	let mut initial_w: Matrix<f64> = Matrix::new(1, 3);
	initial_w.set(0, 0, 1.0).unwrap();
	initial_w.set(0, 1, 1.0).unwrap();
	initial_w.set(0, 2, 1.0).unwrap();

	let result = simple_lasso(&test_data, &test_result, &initial_w, 0.005, 0.0, 10000).unwrap();
	assert!(result.get(0, 0).unwrap().abs() < 0.005);
	assert!((result.get(0, 1).unwrap() - 1.0).abs() < 0.005);
	assert!((result.get(0, 2).unwrap() - 2.0).abs() < 0.005);
}

#[test]
fn simple_lasso_start_not_solution() {
	let mut test_data: Matrix<f64> = Matrix::new(3, 3);
	test_data.set(1, 0, 1.0).unwrap();
	test_data.set(2, 0, 2.0).unwrap();
	test_data.set(0, 1, 3.0).unwrap();
	test_data.set(1, 1, 4.0).unwrap();
	test_data.set(2, 1, 5.0).unwrap();
	test_data.set(0, 2, 6.0).unwrap();
	test_data.set(1, 2, 7.0).unwrap();
	test_data.set(2, 2, 8.0).unwrap();

	let mut test_result: Matrix<f64> = Matrix::new(1, 3);
	test_result.set(0, 0, 5.0).unwrap();
	test_result.set(0, 1, 14.0).unwrap();
	test_result.set(0, 2, 23.0).unwrap();

	let mut initial_w: Matrix<f64> = Matrix::new(1, 3);
	initial_w.set(0, 0, 1.0).unwrap();
	initial_w.set(0, 1, 1.0).unwrap();
	initial_w.set(0, 2, 1.0).unwrap();

	let result = simple_lasso(&test_data, &test_result, &initial_w, 0.005, 0.001, 10000).unwrap();
	// makes non relevant term exactly 0 but others not as accurate compared to normal regression
	assert_eq!(result.get(0, 0).unwrap(), 0.0);
	assert!((result.get(0, 1).unwrap() - 1.0).abs() < 0.1);
	assert!((result.get(0, 2).unwrap() - 2.0).abs() < 0.1);
}
