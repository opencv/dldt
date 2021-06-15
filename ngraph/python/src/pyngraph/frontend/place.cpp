// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend_manager.hpp"
#include "frontend_manager/frontend_exceptions.hpp"
#include "frontend_manager/frontend_manager.hpp"
#include "pyngraph/function.hpp"

namespace py = pybind11;

void regclass_pyngraph_Place(py::module m)
{
    py::class_<ngraph::frontend::Place, std::shared_ptr<ngraph::frontend::Place>> place(
        m, "Place", py::dynamic_attr());
    place.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";

    place.def("is_input",
              &ngraph::frontend::Place::is_input,
              R"(
                Returns true if this place is input for a model

                Returns
                ----------
                is_input : bool
                    True if this place is input for a model
             )");

    place.def("is_output",
              &ngraph::frontend::Place::is_output,
              R"(
                Returns true if this place is output for a model

                Returns
                ----------
                is_output : bool
                    True if this place is output for a model
             )");

    place.def("get_names",
              &ngraph::frontend::Place::get_names,
              R"(
                All associated names (synonyms) that identify this place in the graph in a framework specific way

                Returns
                ----------
                get_names : List[str]
                    A vector of strings each representing a name that identifies this place in the graph.
                    Can be empty if there are no names associated with this place or name cannot be attached
             )");

    place.def("is_equal",
              &ngraph::frontend::Place::is_equal,
              py::arg("other"),
              R"(
                Returns true if another place is the same as this place.

                Parameters
                ----------
                other : Place
                    Another place object

                Returns
                ----------
                is_equal : bool
                    True if another place is the same as this place
             )");

    place.def("is_equal_data",
              &ngraph::frontend::Place::is_equal_data,
              py::arg("other"),
              R"(
                Returns true if another place points to the same data
                Note: The same data means all places on path:
                      output port -> output edge -> tensor -> input edge -> input port

                Parameters
                ----------
                other : Place
                    Another place object

                Returns
                ----------
                is_equal_data : bool
                    True if another place points to the same data
             )");

    place.def(
        "get_consuming_operations",
        static_cast<std::vector<ngraph::frontend::Place::Ptr> (ngraph::frontend::Place::*)() const>(
            &ngraph::frontend::Place::get_consuming_operations),
        R"(
                Returns references to all operation nodes that consume data from this place
                Note: It can be called for any kind of graph place searching for the first consuming operations

                Returns
                ----------
                get_consuming_operations : List[Place]
                    A list with all operation node references that consumes data from this place
             )");

    place.def("get_consuming_operations",
              static_cast<std::vector<ngraph::frontend::Place::Ptr> (ngraph::frontend::Place::*)(
                  int) const>(&ngraph::frontend::Place::get_consuming_operations),
              py::arg("outputPortIndex"),
              R"(
                Returns references to all operation nodes that consume data from this place for specified output port
                Note: It can be called for any kind of graph place searching for the first consuming operations

                Parameters
                ----------
                outputPortIndex : int
                    If place is an operational node it specifies which output port should be considered

                Returns
                ----------
                get_consuming_operations : List[Place]
                    A list with all operation node references that consumes data from this place
             )");

    place.def("get_target_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_target_tensor),
              R"(
                Returns a tensor place that gets data from this place; applicable for operations,
                output ports and output edges

                Parameters
                ----------
                outputPortIndex : int
                    Output port index if the current place is an operation node and has multiple output ports

                Returns
                ----------
                get_consuming_operations : Place
                    A tensor place which hold the resulting value for this place
             )");

    place.def("get_target_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_target_tensor),
              py::arg("outputPortIndex"),
              R"(
                Returns a tensor place that gets data from this place; applicable for operations,
                output ports and output edges which have only one output port

                Returns
                ----------
                get_consuming_operations : Place
                    A tensor place which hold the resulting value for this place
             )");

    place.def("get_producing_operation",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_producing_operation),
              R"(
                Get an operation node place that immediately produces data for this place; applicable if place
                has only one input port

                Returns
                ----------
                get_producing_operation : Place
                    An operation place that produces data for this place
             )");

    place.def("get_producing_operation",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_producing_operation),
              py::arg("inputPortIndex"),
              R"(
                Get an operation node place that immediately produces data for this place

                Parameters
                ----------
                inputPortIndex : int
                    If a given place is itself an operation node, this specifies a port index

                Returns
                ----------
                get_producing_operation : Place
                    An operation place that produces data for this place
             )");

    place.def("get_producing_port",
              &ngraph::frontend::Place::get_producing_port,
              R"(
                Returns a port that produces data for this place

                Returns
                ----------
                get_producing_port : Place
                    A port place that produces data for this place
             )");

    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_input_port),
              R"(
                For operation node returns reference to an input port; applicable if operation node
                has only one input port

                Returns
                ----------
                get_input_port : Place
                    Input port place
             )");

    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_input_port),
              py::arg("inputPortIndex"),
              R"(
                For operation node returns reference to an input port with specified index

                Parameters
                ----------
                inputPortIndex : int
                    Input port index

                Returns
                ----------
                get_input_port : Place
                    Appropriate input port place
             )");

    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&) const>(&ngraph::frontend::Place::get_input_port),
              py::arg("inputName"),
              R"(
                For operation node returns reference to an input port with specified name; applicable if
                port group has only one input port

                Parameters
                ----------
                inputName : str
                    Name of port group

                Returns
                ----------
                get_input_port : Place
                    Appropriate input port place
             )");

    place.def("get_input_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&, int) const>(&ngraph::frontend::Place::get_input_port),
              py::arg("inputName"),
              py::arg("inputPortIndex"),
              R"(
                For operation node returns reference to an input port with specified name and index

                Parameters
                ----------
                inputName : str
                    Name of port group

                inputPortIndex : int
                    Input port index in a group

                Returns
                ----------
                get_input_port : Place
                    Appropriate input port place
             )");

    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_output_port),
              R"(
                For operation node returns reference to an output por; applicable for
                operations with only one output port

                Returns
                ----------
                get_output_port : Place
                    Appropriate output port place
             )");

    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_output_port),
              py::arg("outputPortIndex"),
              R"(
                For operation node returns reference to an output por; applicable for
                operations with only one output port

                Parameters
                ----------
                outputPortIndex : int
                    Output port index

                Returns
                ----------
                get_output_port : Place
                    Appropriate output port place
             )");

    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&) const>(&ngraph::frontend::Place::get_output_port),
              py::arg("outputName"),
              R"(
                For operation node returns reference to an output port; applicable for
                operations with only one output port in a group

                Parameters
                ----------
                outputName : str
                    Name of output port group

                Returns
                ----------
                get_output_port : Place
                    Appropriate output port place
             )");

    place.def("get_output_port",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(
                  const std::string&, int) const>(&ngraph::frontend::Place::get_output_port),
              py::arg("outputName"),
              py::arg("outputPortIndex"),
              R"(
                For operation node returns reference to an output port with specified name and index

                Parameters
                ----------
                outputName : str
                    Name of output port group

                outputPortIndex : int
                    Output port index

                Returns
                ----------
                get_output_port : Place
                    Appropriate output port place
             )");

    place.def("get_consuming_ports",
              &ngraph::frontend::Place::get_consuming_ports,
              R"(
                Returns all input ports that consume data flows through this place

                Returns
                ----------
                get_consuming_ports : List[Place]
                    Input ports that consume data flows through this place
             )");

    place.def("get_source_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)() const>(
                  &ngraph::frontend::Place::get_source_tensor),
              R"(
                Returns a tensor place that supplies data for this place; applicable for operations,
                input ports and input edges which have only one input port

                Returns
                ----------
                get_source_tensor : Place
                    A tensor place which supplies data for this place
             )");

    place.def("get_source_tensor",
              static_cast<ngraph::frontend::Place::Ptr (ngraph::frontend::Place::*)(int) const>(
                  &ngraph::frontend::Place::get_source_tensor),
              py::arg("inputPortIndex"),
              R"(
                Returns a tensor place that supplies data for this place; applicable for operations,
                input ports and input edges

                Parameters
                ----------
                inputPortIndex : int
                    Input port index for operational nodes

                Returns
                ----------
                get_source_tensor : Place
                    A tensor place which supplies data for this place
             )");
}
