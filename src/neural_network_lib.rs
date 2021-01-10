pub mod matrix;
use matrix::*;
use std::fs;

pub struct neural_network {
    pub num_layers: usize,
    pub shape: Vec<usize>,
    pub weights: Vec<Matrix>,
}

impl neural_network {
    pub fn new() -> Self{
        neural_network{
            num_layers: 0,
            shape: Vec::new(),
            weights: Vec::new()
        }
    }

    pub fn init(&mut self, shape: Vec<usize>){
        self.num_layers = shape.len() -1;
        self.shape = shape[0..shape.len()].to_vec();

        for i in 0..shape.len() - 1{
            self.weights.push( Matrix::new( shape[i+1] ,shape[i] ) );
        }   
    }

    pub fn init_with_file(&mut self, filename: &str){
        
        let contents = fs::read_to_string(filename.to_string())
            .expect("Something went wrong reading the file");

        let data_vec: Vec<&str> = contents.split('\n').collect();

        for i in data_vec[0].split(' '){
            self.shape.push( i.parse::<usize>().unwrap() );
        }
            
        self.num_layers = self.shape.len() -1;

        for i in 0..self.num_layers{
            self.weights.push( Matrix::new( self.shape[i+1] ,self.shape[i] ) );
        } 

        for i in 1..data_vec.len(){
            let weight_data : Vec<&str> = data_vec[i].split(' ').collect();
            let mut idx = 0;
            for row in 0..self.weights[i-1].rows_num{
                for col in 0..self.weights[i-1].columns_num{
                    self.weights[i-1].data[row][col] = weight_data[idx].parse::<f64>().unwrap();
                    idx+=1;
                }
            }
        }
    }

    pub fn train(&mut self, filename: &str){
        let contents = fs::read_to_string(filename.to_string())
        .expect("Something went wrong reading the file");
        let data_vec: Vec<&str> = contents.split('\n').collect();
        
        for iteration in 0..10000000{
            let index = (rand::random::<f32>() * data_vec.len() as f32).floor() as usize;
            let current_line : Vec<&str> = data_vec[index].split('|').collect();
            //let current_line : Vec<&str> = data_vec[1].split('|').collect();

            let input_string = current_line[0];
            let target_string = current_line[1];

            let mut input_matrix = Matrix::new(self.shape[0], 1);
            let mut target_matrix = Matrix::new(self.shape[self.shape.len()-1], 1);

            let mut idx = 0;
            for i in input_string.split(' '){
                input_matrix.data[idx][0] = i.parse::<f64>().unwrap();
                idx+=1;
            }

            idx = 0;
            for i in target_string.split(' '){
                target_matrix.data[idx][0] = i.parse::<f64>().unwrap();
                idx+=1;
            }

            let mut error_matrix = Vec::<Matrix>::with_capacity(self.num_layers);
            let output = self.feed_forward(&input_matrix);
            error_matrix.push(subtract_matrix(&target_matrix, &output));

            let mut err_idx = 0;

            for i in self.weights.iter().rev(){
                //error_matrix[err_idx].print_data();
                error_matrix.push(dot_product(&i.transpose(), &error_matrix[err_idx]));
                err_idx += 1;
            }
            error_matrix.pop();
            // for i in 0..error_matrix.len(){
            //     //i = &i.square();
            //     error_matrix[i].print_data();
            // }

            let mut err_layer = Vec::<f64>::new();
            for i in 0..self.num_layers{
                //println!("LAYER NOOOOOOOOOOOOOOOOOO {}", i);
                err_layer.push(error_matrix[i].sum_all());
            }

            for i in 0..err_layer.len(){
                //println!("ERROOOOOOOOR {}",i);
                err_layer[i] = err_layer[i]*err_layer[i];
            }
            //delta W = Lr * ErrMatrix *-element multiplication-* dsigmoid(output) * H-out_transpose

            for layer_idx in (1..self.num_layers+1).rev(){// this is the backpropagation loop
                //println!("LAYER INDEX  {}", layer_idx);
                let mut hidden_out =  self.layer_output(&input_matrix, &layer_idx);
                // hidden_out.print_data();
                // error_matrix[self.num_layers - layer_idx].print_data();
                //let mut delta_w = element_multiply(&error_matrix[self.num_layers - layer_idx], &hidden_out.dsigmoid());
                let mut delta_w = hidden_out.dsigmoid().scalar_multiply(err_layer[self.num_layers - layer_idx]);
                //delta_w.print_data();
                //delta_w = delta_w.scalar_multiply(-0.2);
                delta_w = dot_product(&delta_w, &self.layer_output(&input_matrix, &(layer_idx-1)).transpose());
                //self.layer_output(&input_matrix, &(layer_idx-1)).print_data();
                //hidden_out.print_data();
                //delta_w.print_data();
                //self.weights[layer_idx-1].print_data();
                delta_w = delta_w.scalar_multiply(0.2);
                self.weights[layer_idx-1]= add_matrix(&self.weights[layer_idx-1], &delta_w);
                //self.weights[layer_idx].print_data();

            }
            if iteration % 10000 == 0 {

                println!("{}", err_layer[0]);
            }

        }
    }

    pub fn save_to_file(&self, filename: &str){
        let mut output = String::new();

        for i in 0..self.shape.len(){
            output.push_str(&self.shape[i].to_string());
            if i != self.num_layers{
                output.push(' ');
            }
        }
        output.push('\n');
        for i in 0..self.num_layers{
            for row in 0..self.weights[i].rows_num{
                for col in 0..self.weights[i].columns_num{
                    output.push_str(&self.weights[i].data[row][col].to_string());
                    if row + col !=  self.weights[i].rows_num + self.weights[i].columns_num-2{
                        output.push(' ');
                    }
                }
            }
            if i != self.num_layers-1{
                output.push('\n');
            }
        } 
        #[allow(unused_must_use)]
        fs::write(filename, output);
    }



    pub fn randomise_weights(&mut self){
        for i in 0..self.num_layers{
            self.weights[i].randomise();
        }        
    }

    pub fn show_details(&self){
        println!("-----------------------");
        for i in &self.weights{
            i.print_data();
            println!("");
            println!("");
        }
        print!("Shape  ");
        for i in 0..self.num_layers+1{
            print!("{}  ",self.shape[i]);
        }
        println!("");
        println!("num_layers = {}", self.num_layers);
        println!("-----------------------");
    }

    pub fn feed_forward(&self, input_matrix: &Matrix)-> Matrix{
        //let mut input_matrix = Matrix::new(self.shape[0], 1);
        //input_matrix.from_array(&inputs);
        let mut temp = dot_product(&self.weights[0], &input_matrix).sigmoid();
        for i in 1..self.num_layers{
            temp = dot_product(&self.weights[i], &temp).sigmoid(); 
            
        }
        temp
    }

    pub fn layer_output(&self, input_matrix: &Matrix, layer_idx: &usize)-> Matrix{
        if *layer_idx == 0{
            let mut inp = input_matrix.copy();
            inp
        }
        else{
        let mut temp = dot_product(&self.weights[0], &input_matrix).sigmoid();
        for i in 1..*layer_idx{
            temp = dot_product(&self.weights[i], &temp).sigmoid();    
        }
        temp
    }
    }
}