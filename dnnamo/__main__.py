#!/usr/bin/env python
import sys
import argparse

import importlib

from .tools import ToolRegistry

def main(argv):
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Tools', description='Run each tool with -h or --help for more info.')

  for toolname,toolcls in ToolRegistry.sorted_tools():
    tool = toolcls()
    subparser = tool.add_subparser(subparsers)
    subparser.set_defaults(selected_tool=tool.run)

  args = parser.parse_args(argv)
  args.selected_tool(vars(args))

if __name__=='__main__':
  main(sys.argv[1:])
