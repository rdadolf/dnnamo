#!/usr/bin/env python
import sys
import argparse

import importlib

# NOTE: These are filenames, not the command-line names.
#       Command-line names are specified by the Tool.TOOL_NAME property.
tools = [
  'amdahl',
  'dendrogram',
  'native_ops',
  'native_op_distribution',
  'native_op_profile',
  'native_op_density',
  'native_op_breakdown',
  'primops',
  'ubench_stats',
  # Internal tools
  '_primop_diag',
]


def main(argv):
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Tools', description='Run each tool with -h or --help for more info.')
  tooldict = {t:None for t in tools}
  for toolname in tools:
    # FIXME: maybe switch to using imp import of importlib here
    # FIXME: add a Tool abstract base class and subclass all tools from it
    # FIXME: use inspect to load all classes which inherit from the Tool ABC
    tooldict[toolname] = importlib.import_module(toolname).Tool()
    subparser = tooldict[toolname].add_subparser(subparsers)
    subparser.set_defaults(selected_tool=tooldict[toolname].run)

  args = parser.parse_args(argv)
  args.selected_tool(vars(args))

if __name__=='__main__':
  main(sys.argv[1:])
