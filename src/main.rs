#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
//#![allow(unused_variables)]
#![allow(unused_must_use)]

pub mod neural_network_lib;
use neural_network_lib::*;

fn main() {

    let mut nn = neural_network::new();
    nn.init(vec![2,3,1]);
    //nn.init_with_file("testsave.txt");
    //nn.init(vec![50*50, 20*20, 8*8]);
    nn.randomise_weights();
    //nn.show_details();
    //nn.save_to_file("testsave.txt");

    nn.train("trainingdata.txt");

}
