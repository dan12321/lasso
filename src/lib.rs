pub mod linalg;
use linalg::{Matrix, Error};

/// Makes a step that reduces the MSE
///
/// # Arguments
///
/// * `data` - The training data
/// * `solution` - The test data
/// * `w` - The current map from data to solution
/// * `step_length` - How far to travel in the direction of gradient decent
/// 
/// # Returns
/// 
/// The new 'w' value in the direction of gradient decent
fn linear_regression_step(data: &Matrix<f64>, solution: &Matrix<f64>, w: &Matrix<f64>, step_length: f64) -> Result<Matrix<f64>, Error> {
	let gradient = match mse_gradient(data, solution, w) {
		Ok(g) => g,
		Err(e) => return Err(e),
	};
	let step = gradient.mul_scalar(-2.0 * step_length);
	w.add(&step)
}

fn mse_gradient(data: &Matrix<f64>, solution: &Matrix<f64>, w: &Matrix<f64>) -> Result<Matrix<f64>, Error> {
	let height = data.height();
	let mut sum = Matrix::new(w.width(), w.height());
	for i in 0..height {
		let row = match data.row(i) {
			Ok(r) => r,
			Err(e) => return Err(e)
		};
		let solution_element = match solution.get(0, i) {
			Ok(element) => element,
			Err(e) => return Err(e)
		};
		let coef = match row.dot(w) {
			Ok(scalar) => scalar - solution_element,
			Err(e) => return Err(e)
		};
		let value = row.mul_scalar(coef);
		sum = match sum.add(&value) {
			Ok(vector) => vector,
			Err(e) => return Err(e)
		};
	}
	Ok(sum.mul_scalar(1.0 / height as f64))
}

/// Pull the values of a matrix towards 0
///
/// # Arguments
///
/// * `matrix` - The starting matrix
/// * `lambda` - The amount to pull the elements by
/// 
/// # Returns
/// 
/// A new matrix with values pulled towards 0
fn soft_threshold(matrix: &Matrix<f64>, lambda: f64) -> Result<Matrix<f64>, Error> {
	let mut thresholded_matrix = Matrix::new(matrix.width(), matrix.height());
	for i in 0..matrix.width() {
		for j in 0..matrix.height() {
			let thresholded_element = match matrix.get(i, j) {
				Ok(element) => match element {
					e if e > lambda => e - lambda,
					e if e < -lambda => e + lambda,
					_ => 0.0,
				},
				Err(e) => return Err(e)
			};
			match thresholded_matrix.set(i, j, thresholded_element) {
				Ok(_) => (),
				Err(e) => return Err(e)
			};
		}
	}
	Ok(thresholded_matrix)
}

/// Makes a step that reduces the MSE + 1-norm
///
/// # Arguments
///
/// * `data` - The training data
/// * `solution` - The test data 
/// * `w` - The current map from data to solution
/// * `step_length` - How far to travel in the direction of gradient decent for MSE
/// * `lambda` - The penalty amount of the 1-norm of w
/// 
/// # Returns
/// 
/// The new 'w' value in the direction reducing MSE + 1-norm (might not reduce if it overshoots)
fn lasso_step(data: &Matrix<f64>, solution: &Matrix<f64>, w: &Matrix<f64>, step_length: f64, lambda: f64) -> Result<Matrix<f64>, Error> {
	let linear_regression_w = match linear_regression_step(data, solution, w, step_length) {
		Ok(vector) => vector,
		Err(e) => return Err(e),
	};
	soft_threshold(&linear_regression_w, lambda)
}

/// Performs lasso regression
///
/// # Arguments
///
/// * `data` - The training data
/// * `solution` - The test data 
/// * `initial_w` - The starting map from data to solution
/// * `step_length` - How far to travel in the direction of gradient decent for MSE
/// * `lambda` - The penalty amount of the 1-norm of w
/// * `steps` - The number of steps to take
/// 
/// # Returns
/// 
/// The new 'w' value after the regression has been performed 
pub fn simple_lasso(data: &Matrix<f64>, solution: &Matrix<f64>, initial_w: &Matrix<f64>, step_length: f64, lambda: f64, steps: usize) -> Result<Matrix<f64>, Error> {
	let mut w = initial_w.clone();
	for _ in 0..steps {
		w = match lasso_step(data, solution, &w, step_length, lambda) {
			Ok(new_w) => new_w,
			Err(e) => return Err(e),
		}
	}
	Ok(w)
}
