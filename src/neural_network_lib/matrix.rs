//This is a very simple matrix library
use rand::prelude::*;

pub struct Matrix{
    pub rows_num: usize,
    pub columns_num: usize,

    pub data: Vec<Vec<f64>>,
}


impl Matrix{
    pub fn new(r:usize, c:usize) -> Self{
        Matrix{
            rows_num: r,
            columns_num: c,
            data: vec![vec![0.0;c];r],
        }
    }

    pub fn copy(&self)-> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);

        for row in 0..output.rows_num{
            for col in 0..output.columns_num{
                output.data[row][col] = self.data[row][col];
            }
        }
        output
    }

    pub fn print_data(&self){
        println!("Rows: {}", self.rows_num);
        println!("Columns: {}", self.columns_num);
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                print!("{}  ", self.data[row][col]);
            }
            println!("");      
       }
       println!("");
       println!("");
    }

    pub fn randomise(&mut self){
        let mut rng = rand::thread_rng();
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                self.data[row][col] = rng.gen() ;
            }
        }
    }

    pub fn from_array(&mut self, a: &[&[f64]]){
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                self.data[row][col] = a[row][col];
            }  
       }   
    }

    pub fn from_vec(&mut self, a: Vec<Vec<f64>>){
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                self.data[row][col] = a[row][col];
            }  
       }   
    }

    pub fn scalar_add(&self, inc:f64) -> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[row][col] = self.data[row][col] + inc ;
            }
        }
        output
    }

    pub fn sum_all(&self) -> f64{
        if self.columns_num != 1{
            panic!("CANNOT CALCULATE ERROR");
        }
        let mut output = 0.0;
        for row in 0..self.rows_num{
                output += self.data[row][0] ;
        }
        output
    }

    pub fn transpose(&self) -> Self{
        let mut output = Matrix::new(self.columns_num, self.rows_num);
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[col][row] = self.data[row][col];
            }
        }
        output
    }

    pub fn scalar_multiply(&mut self, inc:f64) -> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[row][col] = self.data[row][col] * inc ;
            }
        }
        output
    }

    pub fn square(&mut self) -> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[row][col] = self.data[row][col] * self.data[row][col] ;
            }
        }
        output
    }

    pub fn sigmoid(&mut self) -> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);
        fn s(x: &f64)-> f64{
            1.0 / ( 1.0 + x.exp() )
        }
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[row][col] = s(&self.data[row][col]);
            }
        }
        output
    }

    pub fn dsigmoid(&mut self) -> Self{
        let mut output = Matrix::new(self.rows_num, self.columns_num);
        fn ds(x: &f64)-> f64{
            x*(1.0-x)
        }
        for row in 0..self.rows_num{
            for col in 0..self.columns_num{
                output.data[row][col] = ds(&self.data[row][col]);
            }
        }
        output
    }
} 


pub fn add_matrix(a: &Matrix, b: &Matrix) -> Matrix{
    if a.rows_num != b.rows_num || a.columns_num != b.columns_num {
        panic!("The matrices cannot be added.");
    }
    let mut sum = Matrix::new(a.rows_num, a.columns_num);
    for row in 0..a.rows_num{
        for col in 0..a.columns_num{
            sum.data[row][col] = a.data[row][col] + b.data[row][col];
        }
    }
    sum
}

pub fn element_multiply(a: &Matrix, b: &Matrix) -> Matrix{
    if a.rows_num != b.rows_num || a.columns_num != b.columns_num {
        panic!("The matrices cannot be multplied(element).");
    }
    let mut product = Matrix::new(a.rows_num, a.columns_num);
    for row in 0..a.rows_num{
        for col in 0..a.columns_num{
            product.data[row][col] = a.data[row][col] * b.data[row][col];
        }
    }
    product
}

pub fn subtract_matrix(a: &Matrix, b: &Matrix) -> Matrix{
    if a.rows_num != b.rows_num || a.columns_num != b.columns_num {
        panic!("The matrices cannot be added.");
    }
    let mut sum = Matrix::new(a.rows_num, a.columns_num);
    for row in 0..a.rows_num{
        for col in 0..a.columns_num{
            sum.data[row][col] = a.data[row][col] - b.data[row][col];
        }
    }
    sum
}

pub fn dot_product(a: &Matrix, b: &Matrix) -> Matrix{
    if a.columns_num != b.rows_num {
        panic!("The matrices cannot be multiplied.");
    }
    let mut product = Matrix::new(a.rows_num, b.columns_num);
    for row in 0..product.rows_num{
        for col in 0..product.columns_num{   
            for idx in 0..a.columns_num{
                product.data[row][col] += a.data[row][idx] * b.data[idx][col];
            }
        }
    }
    product
}
