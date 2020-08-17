#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"

#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;

using namespace InferenceEngine::HeteroConfigParams;



// ...

auto execNetwork = ie.LoadNetwork(network, "HETERO:FPGA,CPU", { {KEY_HETERO_DUMP_GRAPH_DOT, YES} });

return 0;
}
