// nnet3/nnet-nnet.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_NNET_H_
#define KALDI_NNET3_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-descriptor.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

namespace kaldi {
namespace nnet3 {

enum NodeType { kInput, kDescriptor, kComponent, kDimRange, kNone };


/// NetworkNode is used to represent, three types of thing: either an input of the
/// network (which pretty much just states the dimension of the input vector);
/// a Component (e.g. an affine component or a sigmoid component); or a Descriptor.
/// A Descriptor is basically an expression that can do things like append
/// the outputs of other components (or inputs) together, add them together, and
/// do various other things like shifting the time index.
///
/// Each Component must have an input of type kDescriptor that is numbered
/// Preceding to the Component, and that is not used elsewhere.  This may seem
/// unintuitive but it makes the implementation a lot easier; any apparent waste
/// can be optimized out after compilation.  And outputs must also be of type
/// kDescriptor.
///
/// Note: in the actual computation you can provide input not only to nodes of
/// type kInput but also to nodes of type kComponent; this is useful in things
/// like recurrent nets where you may want to split the computation up into
/// pieces.
///
/// Note that in the config-file format, there are three types of node: input,
/// component and output.  output maps to kDescriptor, but the nodes of type
/// kDescriptor that represent the input to a component, are described in the
/// same config-file line as the Component itself.
struct NetworkNode {  
  NodeType node_type;
  // "descriptor" is relevant only for nodes of type kDescriptor.
  Descriptor descriptor;
  union {
    // For kComponent, the index into Nnet::components_
    int32 component_index;
    // for kDimRange, the node-index of the input node, which must be of
    // type kComponent or kInput.
    int32 node_index;
  } u;
  // for kInput, the dimension of the input feature.  For kDimRange, the dimension
  // of the output (i.e. the length of the range)
  int32 dim;
  // for kDimRange, the dimension of the offset into the input component's feature.
  int32 dim_offset;
  
  int32 Dim(const Nnet &nnet) const;  // Dimension that this node outputs.

  NetworkNode(NodeType nt = kNone):
      node_type(nt), dim(-1), dim_offset(-1) { u.component_index = -1; }
  NetworkNode(const NetworkNode &other);  // copy constructor.
  // use default assignment operator
};



class Nnet {
 public:
  // This function can be used either to initialize a new Nnet from a config
  // file, or to add to an existing Nnet, possibly replacing certain parts of
  // it.  It will die with error if something went wrong.
  void ReadConfig(std::istream &config_file);
  
  int32 NumComponents() const { return components_.size(); }

  int32 NumNodes() const { return nodes_.size(); }
  
  /// return component indexed c.  not a copy; not owned by caller.
  Component *GetComponent(int32 c);

  /// return component indexed c (const version).  not a copy; not owned by
  /// caller.
  const Component *GetComponent(int32 c) const;


  /// return the component corresponding to the node indexed n, which must
  /// be of type kComponent.  Convenience function.  Result is not a copy and
  /// not owned by the caller.
  Component *GetComponentForNode(int32 n);
  /// Const version of GetComponentForNode().
  const Component *GetComponentForNode(int32 n) const;



  /// returns const reference to a particular numbered network node.
  const NetworkNode &GetNode(int32 node) const {
    KALDI_ASSERT(node >= 0 && node < nodes_.size());
    return nodes_[node];
  }

  /// Returns true if this is a component node, meaning that it is of type
  /// kComponent.
  bool IsComponentNode(int32 node) const;

  /// Returns true if this is a dim-range node, meaning that it is of type
  /// kDimRange.
  bool IsDimRangeNode(int32 node) const;
  
  /// Returns true if this is an output node, meaning that it is of type
  /// kInput.
  bool IsInputNode(int32 node) const;

  /// Returns true if this is a descriptor node, meaning that it is of type
  /// kDescriptor.  Exactly one of IsOutput or IsComponentInput will also
  /// apply.
  bool IsDescriptorNode(int32 node) const;
  
  /// Returns true if this is an output node, meaning that it is of type kDescriptor
  /// and is not directly followed by a node of type kComponent.
  bool IsOutputNode(int32 node) const;

  /// Returns true if this is component-input node, i.e. a node of type kDescriptor
  /// that immediately precedes a node of type kComponent.
  bool IsComponentInputNode(int32 node) const;  

  /// returns vector of node names (needed by some parsing code, for instance).
  const std::vector<std::string> &GetNodeNames() const;

  /// returns individual node name.
  const std::string &GetNodeName(int32 node_index) const;

  /// returns vector of component names (needed by some parsing code, for instance).
  const std::vector<std::string> &GetComponentNames() const;

  /// returns individual component name.
  const std::string &GetComponentName(int32 component_index) const;
  
  /// returns index associated with this node name, or -1 if no such index.
  int32 GetNodeIndex(const std::string &node_name) const;

  /// returns index associated with this component name, or -1 if no such index.
  int32 GetComponentIndex(const std::string &node_name) const;

  // This convenience function returns the dimension of the input with name
  // "input_name" (e.g. input_name="input" or "ivector"), or -1 if there is no
  // such input.
  int32 InputDim(const std::string &input_name) const;

  // This convenience function returns the dimension of the output with
  // name "input_name" (e.g. output_name="input"), or -1 if there is
  // no such input.
  int32 OutputDim(const std::string &output_name) const;
  
  void Read(std::istream &istream, bool binary);

  void Write(std::ostream &ostream, bool binary) const;

  /// note to self: one thing of many that we need to check is that no output
  /// nodes are referred to in Descriptors.  This might mess up the combination
  /// of each output node into a single step, as dependencies would be messed
  /// up.  Also make sure no nodes referred to in Descriptors, or in kDimRange,
  /// are themselves Descriptors.
  void Check() const;

  /// returns some human-readable information about the network, mostly for
  /// debugging purposes.
  std::string Info() const;


  /// [Relevant for clockwork RNNs and similar].  Computes the smallest integer
  /// n >=1 such that the neural net's behavior will be the same if we shift the
  /// input and output's time indexes (t) by integer multiples of n.  Does this
  /// by computing the lcm of all the moduli of the Descriptors in the network.
  int32 Modulus() const;

  ~Nnet() { Destroy(); }
 private:
  void Destroy();
  
  // This function returns as a string the contents of a line of a config-file
  // corresponding to the node indexed "node_index", which must not be of type
  // kComponentInput, in the same format as it would appear in a line of a
  // config-file.
  std::string GetAsConfigLine(int32 node_index) const;

  // This function outputs to "config_lines" the lines of a config file, which
  // will be sufficient to reconstruct the nodes in the network (but not
  // the components, which need to be written separately).
  void GetConfigLines(std::vector<std::string> *config_lines) const;

  // This function is used when reading config files; it exists in order to
  // handle replacement of existing nodes.  The two input vectors have the same
  // size.  Its job is to remove redundant lines that do not have "component" as
  // first_token, and where two lines have a configuration value name=xxx in the
  // config with the same name.  In this case it removes the first of the two,
  // but that first one must have index less than num_lines_initial, else it is
  // an error.
  // This function also checks that all lines have a config name=xxx, that
  // IsValidName(xxx) is true, and that there are no two lines with "component"
  // as the first token and with the same config name=xxx.  Note: here, "name"
  // means literally "name", but "xxx" stands in for the actual name,
  // e.g. "my-funky-component."
  static void RemoveRedundantConfigLines(int32 num_lines_initial,
                                         std::vector<std::string> *first_tokens,
                                         std::vector<ConfigLine> *configs);
  
  void ProcessComponentConfigLine(int32 initial_num_components,
                                  ConfigLine *config);
  void ProcessComponentNodeConfigLine(int32 pass,
                                      ConfigLine *config);
  void ProcessInputNodeConfigLine(ConfigLine *config);
  void ProcessOutputNodeConfigLine(int32 pass,
                                   ConfigLine *config);
  void ProcessDimRangeNodeConfigLine(int32 pass,
                                     ConfigLine *config);

  // This function output to "modified_node_names" a modified copy of
  // node_names_, in which all nodes which are not of type kComponent, kInput or
  // kDimRange are replaced with the string "***".  This is useful when parsing
  // Descriptors, to avoid inadvertently accepting nodes of invalid types where
  // they are not allowed.
  void GetSomeNodeNames(std::vector<std::string> *modified_node_names) const;

  
  // the names of the components of the network.  Note, these may be distinct
  // from the network node names below (and live in a different namespace); the
  // same component may be used in multiple network nodes, to define parameter
  // sharing.
  std::vector<std::string> component_names_;
  
  // the components of the nnet, in arbitrary order.  The network topology is
  // defined separately, below; a given Component may appear more than once in
  // the network if necessary for parameter tying.
  std::vector<Component*> components_;  

  // names of network nodes, i.e. inputs, components and outputs, used only in
  // reading and writing code.  Indexed by network-node index.  Note,
  // components' names are always listed twice, once as foo-input and once as
  // foo, because the input to a component always gets its own NetworkNode index.
  std::vector<std::string> node_names_;

  // the network nodes of the network.
  std::vector<NetworkNode> nodes_;
  
};



} // namespace nnet3
} // namespace kaldi

#endif
