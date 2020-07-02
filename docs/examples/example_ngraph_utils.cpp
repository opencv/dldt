// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

// ! [ngraph:include]
#include <ngraph/opsets/opset3.hpp>
// ! [ngraph:include]

#include <ngraph/function.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/visualize_tree.hpp>

using namespace ngraph;

// ! [ngraph_utils:simple_function]
std::shared_ptr<ngraph::Function> create_simple_function() {
    // This example shows how to create ngraph::Function
    //
    // Parameter--->Multiply--->Add--->Result
    //    Constant---'          /
    //              Constant---'

    // Create opset3::Parameter operation with static shape
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});

    auto mul_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    auto mul = std::make_shared<ngraph::opset3::Multiply>(data, mul_constant);

    auto add_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
    auto add = std::make_shared<ngraph::opset3::Add>(mul, add_constant);

    // Create opset3::Result operation
    auto res = std::make_shared<ngraph::opset3::Result>(mul);

    // Create nGraph function
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:simple_function]

// ! [ngraph_utils:advanced_function]
std::shared_ptr<ngraph::Function> create_advanced_function() {
    // Advanced example with multi output operation
    //
    // Parameter->Split---0-->Result
    //               | `--1-->Relu-->Result
    //               `----2-->Result

    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64, 64});

    // Create Constant for axis value
    auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}/*scalar shape*/, {1});

    // Create opset3::Split operation that splits input to three slices across 1st dimension
    auto split = std::make_shared<ngraph::opset3::Split>(data, axis_const, 3);

    // Create opset3::Relu operation that takes 1st Split output as input
    auto relu = std::make_shared<ngraph::opset3::Relu>(split->output(1)/*specify explicit output*/);

    // Results operations will be created automatically based on provided OutputVector
    return std::make_shared<ngraph::Function>(ngraph::OutputVector{split->output(0), relu, split->output(2)}, ngraph::ParameterVector{data});
}
// ! [ngraph_utils:advanced_function]

void pattern_matcher_examples() {
{
// ! [pattern:simple_example]
// Pattern example
auto input = std::make_shared<ngraph::opset3::Parameter>(element::i64, Shape{1});
auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

// Create Matcher with Parameter->ShapeOf pattern
auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "MyPatternBasedTransformation");
// ! [pattern:simple_example]

// ! [pattern:callback_example]
ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
    // Get root node
    std::shared_ptr<Node> root_node = m.get_match_root();

    // Get all nodes matched by pattern
    NodeVector nodes = m.get_matched_nodes();

    // Transformation code
    return false;
};
// ! [pattern:callback_example]
}

{
// ! [pattern:label_example]
// Detect Multiply with arbitrary first input and second as Constant
// ngraph::pattern::op::Label - represent arbitrary input
auto input = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1});
auto value = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
auto mul = std::make_shared<ngraph::opset3::Multiply>(input, value);
auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "MultiplyMatcher");
// ! [pattern:label_example]
}

{
// ! [pattern:concat_example]
// Detect Concat operation with arbitrary number of inputs
auto concat = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{}, ngraph::pattern::has_class<ngraph::opset3::Concat>());
auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "ConcatMatcher");
// ! [pattern:concat_example]
}

{
// ! [pattern:predicate_example]
// Detect Multiply or Add operation
auto lin_op = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{},
              [](const std::shared_ptr<ngraph::Node> & node) -> bool {
                    return std::dynamic_pointer_cast<ngraph::opset3::Multiply>(node) ||
                    std::dynamic_pointer_cast<ngraph::opset3::Add>(node);
              });
auto m = std::make_shared<ngraph::pattern::Matcher>(lin_op, "MultiplyOrAddMatcher");
// ! [pattern:predicate_example]
}

}

bool ngraph_api_examples(std::shared_ptr<Node> node) {
{
// ! [ngraph:ports_example]
// Let's suppose that node is opset3::Convolution operation
// as we know opset3::Convolution has two input ports (data, weights) and one output port
Input <Node> data = node->input(0);
Input <Node> weights = node->input(1);
Output <Node> output = node->output(0);

// Getting shape and type
auto pshape = data.get_partial_shape();
auto el_type = data.get_element_type();

// Ggetting parent for input port
Output <Node> parent_output;
parent_output = data.get_source_output();

// Another short way to get partent for output port
parent_output = node->input_value(0);

// Getting all consumers for output port
auto consumers = output.get_target_inputs();
// ! [ngraph:ports_example]
}

{
// ! [ngraph:shape]
auto partial_shape = node->input(0).get_partial_shape(); // get zero input partial shape
if (partial_shape.is_dynamic() /* or !partial_shape.is_staic() */) {
    return false;
}
auto static_shape = partial_shape.get_shape();
// ! [ngraph:shape]
}

{
// ! [ngraph:shape_check]
auto partial_shape = node->input(0).get_partial_shape(); // get zero input partial shape

// Check that input shape rank is static
if (!partial_shape.rank().is_static()) {
    return false;
}
auto rank_size = partial_shape.rank().get_length();

// Check that second dimension is not dynamic
if (rank_size < 2 || partial_shape[1].is_dynamic()) {
    return false;
}
auto dim = partial_shape[1].get_length();
// ! [ngraph:shape_check]
}

return true;
}

// ! [ngraph:replace_node]
bool ngraph_replace_node(std::shared_ptr<Node> node) {
    // Step 1. Verify that node has opset3::Negative type
    auto neg = std::dynamic_pointer_cast<ngraph::opset3::Negative>(node);
    if (!neg) {
        return false;
    }

    // Step 2. Create opset3::Multiply operation where the first input is negative operation input and second as Constant with -1 value
    auto mul = std::make_shared<ngraph::opset3::Multiply>(neg->input_value(0),
                                                          opset3::Constant::create(neg->get_element_type(), Shape{1}, {-1}));

    mul->set_friendly_name(neg->get_friendly_name());
    ngraph::copy_runtime_info(neg, mul);

    // Step 3. Replace Negative operation with Multiply operation
    ngraph::replace_node(neg, mul);
    return true;

    // Step 4. Negative operation will be removed automatically because all consumers was moved to Multiply operation
}
// ! [ngraph:replace_node]

// ! [ngraph:insert_node]
// Step 1. Lets suppose that we have a node with single output port and we want to insert additional operation new_node after it
void insert_example(std::shared_ptr<ngraph::Node> node) {
    // Get all consumers for node
    auto consumers = node->output(0).get_target_inputs();

    // Step 2. Create new node. Let it be opset1::Relu.
    auto new_node = std::make_shared<ngraph::opset3::Relu>(node);

    // Step 3. Reconnect all consumers to new_node
    for (auto input : consumers) {
        input.replace_source_output(new_node);
    }
}
// ! [ngraph:insert_node]

// ! [ngraph:insert_node_with_copy]
void insert_example_with_copy(std::shared_ptr<ngraph::Node> node) {
    // Make a node copy
    auto node_copy = node->clone_with_new_inputs(node->input_values());
    // Create new node
    auto new_node = std::make_shared<ngraph::opset3::Relu>(node_copy);
    ngraph::replace_node(node, new_node);
}
// ! [ngraph:insert_node_with_copy]

void eliminate_example(std::shared_ptr<ngraph::Node> node) {
// ! [ngraph:eliminate_node]
// Suppose we have a node that we want to remove
bool success = replace_output_update_name(node->output(0), node->input_value(0));
// ! [ngraph:eliminate_node]
}

// ! [ngraph:visualize]
void visualization_example(std::shared_ptr<ngraph::Function> f) {
    std::vector<std::shared_ptr<ngraph::Function> > g{f};

    // Serialize ngraph::Function to before.svg file before transformation
    ngraph::pass::VisualizeTree("/path/to/file/before.svg").run_on_module(g);

    // Run your transformation
    // ngraph::pass::MyTransformation().run_on_function();

    // Serialize ngraph::Function to after.svg file after transformation
    ngraph::pass::VisualizeTree("/path/to/file/after.svg").run_on_module(g);
}
// ! [ngraph:visualize]