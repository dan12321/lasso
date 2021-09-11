use std::ops::{Add, Sub, Mul};

/// A matrix of numbers
#[derive(Clone)]
pub struct Matrix<T: Add<Output=T> + Mul<Output=T> + Sub<Output=T> + Default> {
	data: Vec<T>,
	stride: usize,
	size: usize,
}

impl<T: Add<Output=T> + Mul<Output=T> + Sub<Output=T> + Copy + Default> Matrix<T> {
	/// Creates a new instance of a matrix using default values
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the matrix
	/// * `height` - The height of the matrix
	pub fn new(width: usize, height: usize) -> Self {
		let size = width * height;
		let stride = width;
		let data: Vec<T> = vec![Default::default(); width * height];
		Matrix {
			data,
			stride,
			size
		}
	}

	/// Creates a new instance of a matrix filled with a value
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the matrix
	/// * `height` - The height of the matrix
	/// * `value` - The height of the matrix
	pub fn new_fill(width: usize, height: usize, value: T) -> Self {
		let size = width * height;
		let stride = width;
		let data: Vec<T> = vec![value; width * height];
		Matrix {
			data,
			stride,
			size
		}
	}

	/// Gets a value at a given position
    ///
    /// # Arguments
    ///
    /// * `i` - The column of the matrix
	/// * `j` - The row of the matrix
	pub fn get(& self, i: usize, j: usize) -> Result<T, Error> {
		let width = self.stride;
		let height = self.size / width;
		if i < width && j < height {
			Ok(self.data[i + j * self.stride])
		} else {
			Err(Error::IndexOutOfRange)
		}
	}

	/// Sets a value at a given position
    ///
    /// # Arguments
    ///
    /// * `i` - The column of the matrix
	/// * `j` - The row of the matrix
	/// * `value` - The value to set it to
	pub fn set(&mut self, i: usize, j: usize, value: T) -> Result<(), Error> {
		let width = self.stride;
		let height = self.size / width;
		if i < width && j < height {
			self.data[i + j * self.stride] = value;
			Ok(())
		} else {
			Err(Error::IndexOutOfRange)
		}
	}

	/// Add another matrix to the current one
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to add
	/// 
	/// # Returns
	/// 
	/// The resulting matrix of the operation
	pub fn add(& self, matrix: &Matrix<T>) -> Result<Matrix<T>, Error> {
		if self.stride == matrix.stride && self.size == matrix.size {
			let data: Vec<T> = self.data.iter().enumerate().map(|(i, x)| *x + matrix.data[i]).collect();
			let result = Matrix {
				data,
				stride: self.stride,
				size: self.size
			};
			Ok(result)
		} else {
			Err(Error::SizeMissMatch)
		}
	}

	/// Times the matrix by a matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to times by
	/// 
	/// # Returns
	/// 
	/// The resulting matrix of the operation
	pub fn mul(& self, matrix: &Matrix<T>) -> Result<Matrix<T>, Error> {
		if self.stride != matrix.size / matrix.stride {
			return Err(Error::SizeMissMatch);
		}
		let self_height = self.size / self.stride;
		let mut result: Matrix<T> = Matrix::new(matrix.stride, self_height);
		for i in 0..matrix.stride {
			for j in 0..self_height {
				let mut element: T = Default::default();
				for k in 0..self.stride {
					element = element + (matrix.get(i, k).unwrap() * self.get(k, j).unwrap());
				}
				if result.set(i, j, element).is_err() {
					return Err(Error::SizeMissMatch);
				};
			}
		}
        Ok(result)
    }

	/// Times the matrix by a scalar
    ///
    /// # Arguments
    ///
    /// * `a` - The scalar to times the elements by
	pub fn mul_scalar(& self, a: T) -> Matrix<T> {
        let data = self.data.iter().map(|x| *x * a).collect();
		let size = self.size;
		let stride = self.stride;
		Matrix {
			data,
			stride,
			size
		}
    }

	/// Gets the dot product between two vectors.
	/// For matrices of the same size will still get the sum
	/// of elements multiplied by the element in the same position
	/// of the second matrix
	/// 
	/// # Arguments
	/// 
	/// * `matrix` - The second vector (or matrix) to calculate the dot product with
	pub fn dot (& self, matrix: &Matrix<T>) -> Result<T, Error> {
		if self.size == matrix.size && self.stride == matrix.stride {
			let result = self.data
				.iter()
				.enumerate()
				.fold(Default::default(), |x: T, (j, y)| x + *y * matrix.data[j]);
			Ok(result)
		} else {
			Err(Error::SizeMissMatch)
		}
	}

	/// Gets a specified row.
	/// Returns as a vector (width 1, height: same as width of matrix)
	pub fn row(& self, i: usize) -> Result<Matrix<T>, Error> {
		let height = self.size / self.stride;
		if i < height {
			let mut data: Vec<T> = Vec::with_capacity(self.stride);
			for i in i * self.stride..(i + 1) * self.stride {
				data.push(self.data[i]);
			}
			let result = Matrix {
				data,
				stride: 1,
				size: self.stride
			};
			Ok(result)
		} else {
			Err(Error::IndexOutOfRange)
		}
	}

	pub fn width(& self) -> usize {
		self.stride
	}

	pub fn height(& self) -> usize {
		self.size / self.stride
	}
}

#[derive(Debug)]
pub enum Error {
	IndexOutOfRange,
	SizeMissMatch,
}