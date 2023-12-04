
import sys
from typing import List
from dataclasses import dataclass
from collections import OrderedDict
from collections import defaultdict
from collections import ChainMap

from lark import Lark, ast_utils, Transformer, v_args
from lark.tree import Meta

from lark import Lark, Tree
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from lark.lexer import TerminalDef, Token

import regex
this_module = sys.modules[__name__]

#
#   Define AST
#
class _Ast(ast_utils.Ast):
    # This will be skipped by create_transformer(), because it starts with an underscore
    pass

class _SelectExpr(_Ast):
    # This will be skipped by create_transformer(), because it starts with an underscore
    pass


@dataclass
class Name(_Ast):
    "Uses WithMeta to include line-number metadata in the meta attribute"
    meta: Meta
    name: str

@dataclass
class TableName(_Ast):
    table_name: str

@dataclass
class Select(_Ast):
    name: Name
    table_name: TableName

@dataclass
class Where(_Ast):
    name: Name


@dataclass
class SetExpr(_Ast, ast_utils.AsList):
    select: Select


class ToAst(Transformer):
    # Define extra transformation functions, for rules that don't correspond to an AST class.

    @v_args(inline=True)
    def start(self, x):
        return x

class CompletionEngine:
    def complete(self, prefix: str) -> regex.Pattern:
        raise NotImplementedError()

    def is_complete(self, prefix: str) -> bool:
        # return self.complete(prefix) == regex.compile('')
        return self.complete(prefix) == regex.compile('')


class LarkCompletionEngine(CompletionEngine):
    def __init__(self, grammar, start_token, allow_ws: bool = True):
        self.parser = Lark(grammar, start=start_token, 
                            # parser='earley',
                            parser='lalr',
                            debug=True,
                           regex=True)
        self.terminal_dict = self.parser._terminals_dict
        self.allow_ws = allow_ws
        self.transformer = ast_utils.create_transformer(this_module, ToAst())

    def complete(self, prefix: str) -> regex.Pattern:
        interactive_parser = self.parser.parse_interactive(prefix)
        token = None
        # print('prefix:', prefix)
        # breakpoint()
        try:
            for token in interactive_parser.parser_state.lexer.lex(
                    interactive_parser.parser_state):
                # print(token)
                interactive_parser.parser_state.feed_token(token)
        except (UnexpectedCharacters, UnexpectedToken):
            pass
        valid_tokens = interactive_parser.accepts()
        # get the regex for the valid tokens
        valid_regex = [f'{self.terminal_dict[t].pattern.to_regexp()}'
                       for t in valid_tokens
                       if t != '$END']

        if valid_regex and self.allow_ws:
            valid_regex.append("\\s+")

        return regex.compile('|'.join(valid_regex))

    def parse(self, text):
        # 抽取出可能出现的column name，并且整理出column和table/alias的关系

        breakpoint()
        tree = self.parser.parse(text)
        # print(tree.pretty())
        # build_children_map(tree)
        print(text)
        # breakpoint()

        table_trees = check_for_table(tree)
        select_branches = [i for i in iter_subtrees_dfs(tree) if i.data == 'select']
        for table_key, (table_name, table_alias, table_tree, table_parent_tree) in table_trees.items():
            print(table_key, table_name, table_alias)
            tree_in_branch_depth = [in_branch(table_parent_tree, i) for i in select_branches]
            other_branch = [select_branches[i] for i in range(len(select_branches)) if tree_in_branch_depth[i] < 0]
            print(tree_in_branch_depth, len(other_branch))

            min_idx_tree_in_branch_depth = min([i for i in tree_in_branch_depth if i > 0])
            min_idx_tree_in_branch_depth = tree_in_branch_depth.index(min_idx_tree_in_branch_depth)
            current_branch = select_branches[min_idx_tree_in_branch_depth]
            
            table_parent_tree = current_branch
            # print(table_parent_tree)
            # print(table_tree)
            # breakpoint()
            column_trees = check_columns_by_table_tree(table_parent_tree, table_tree, other_branch)
            for col_key, (column_name, table_name, table_alias, column_tree) in column_trees.items():
                print(col_key, column_name, table_name, table_alias)
        print(text)
        # breakpoint()

def iter_subtrees_dfs(tree):
    queue = [tree]
    subtrees = OrderedDict()
    for subtree in queue:
        subtrees[id(subtree)] = subtree
        # Reason for type ignore https://github.com/python/mypy/issues/10999
        queue += [c for c in reversed(subtree.children)  # type: ignore[misc]
                    if isinstance(c, Tree) and id(c) not in subtrees]
    del queue
    return reversed(list(subtrees.values()))


def iter_subtrees_dfs_with_depth(tree):
    queue = [(tree, 0)]
    subtrees = OrderedDict()
    for subtree, depth in queue:
        subtrees[id(subtree)] = (subtree, depth)
        # Reason for type ignore https://github.com/python/mypy/issues/10999
        queue += [(c, depth+1) for c in reversed(subtree.children)  # type: ignore[misc]
                    if isinstance(c, Tree) and id(c) not in subtrees]
    del queue
    return reversed(list(subtrees.values()))
    

def iter_subtrees_bfs(tree):
    stack = [tree]
    stack_append = stack.append
    stack_pop = stack.pop
    while stack:
        node = stack_pop()
        if not isinstance(node, Tree):
            continue
        yield node
        for child in reversed(node.children):
            stack_append(child)

        
def get_token_from_tree(tree):
    subtrees = iter_subtrees_dfs(tree)
    tokens = defaultdict(dict)
    for t in subtrees:
        for c in t.children:
            if isinstance(c, Token):
                tokens[id(c)][c.type] = c.value
    return tokens

def get_token_by_rule(tree, rule):
    subtrees = iter_subtrees_dfs(tree)
    tokens = defaultdict(None)
    for t in subtrees:
        for c in t.children:
            if isinstance(c, Tree) \
                and isinstance(c.data, Token) and c.data.type == 'RULE' \
                and c.data.value == rule:
                tokens.update(get_token_from_tree(c))
    return tokens

def get_token_by_name(tree, tree_name):
    subtrees = iter_subtrees_dfs(tree)
    tokens = defaultdict(None)
    for t in subtrees:
        for c in t.children:
            if isinstance(c, Tree) \
                and c.data == tree_name:
                tokens.update(get_token_from_tree(c))
    return tokens
    

def contain_branch(tree, branches):
    subtrees = iter_subtrees_dfs(tree)
    subtrees = [id(i) for i in subtrees]
    for i in branches:
        if id(i) in subtrees:
            return True
    return False

def in_branch(tree, branch):
    subtrees = iter_subtrees_dfs_with_depth(branch)
    for i, depth in subtrees:
        if id(i) == id(tree):
            return depth
    return -1

def check_columns_by_table_tree(tree, table_tree, other_branch):
    queue = [tree]
    column_trees = OrderedDict()
    subtrees = OrderedDict()
    for subtree in queue:
        subtrees[id(subtree)] = subtree
        for child in subtree.children:
            if child is None or isinstance(child, Token) or child == table_tree:
                continue
            if contain_branch(child, other_branch):
                continue
            if isinstance(child.data, Token) and child.data.type == 'RULE' and child.data.value == 'expression':
                tokens = get_token_by_rule(child, 'column_name')
                merged_dict = dict(ChainMap(*(tokens.values())))
                column_name = merged_dict.get('CNAME') # ESCAPED_STRING, backticks_name

                tokens = get_token_by_name(child, 'table_expression')
                merged_dict = dict(ChainMap(*(tokens.values())))
                table_name = merged_dict.get('TABLE_NAME')
                table_alias = merged_dict.get('CNAME') # ESCAPED_STRING, backticks_name
                column_trees[id(child)] = (column_name, table_name, table_alias, child)

                # common_keys = [key for key in columns]
                # common_keys += [key for key in table_expression]
                # common_keys = set(common_keys)
                # for key in common_keys:
                #     column_name = columns[key].get('CNAME') # ESCAPED_STRING, backticks_name
                #     table_name = table_expression[key].get('TABLE_NAME') if key in table_expression else None
                #     table_alias = table_expression[key].get('CNAME') if key in table_expression else None # ESCAPED_STRING, backticks_name
                #     column_trees[key] = (column_name, table_name, table_alias, child)
            else:
                queue += [child]
    return column_trees


def check_for_table(tree):
    queue = [(i, tree) for i in tree.children]
    tables = OrderedDict()
    subtrees = OrderedDict()
    for subtree, ptree in queue:
        subtrees[id(subtree)] = subtree
        # print(subtree)
        # if subtree.data == 'join':
        #     tokens = get_token_from_tree(subtree)
        #     for key in tokens:
        #         table_name = tokens[key].get('TABLE_NAME')
        #         tables[key] = (table_name, subtree, ptree)

        if subtree.data == 'table':
            # breakpoint()

            # tables = get_token_by_rule(subtree, 'table_name')
            # table_alias = get_token_by_rule(subtree, 'alias')

            tokens = get_token_from_tree(subtree)
            # only for "from_item", not work for other sqls that contain "AS"
            merged_dict = dict(ChainMap(*(tokens.values())))
            table_name = merged_dict.get('TABLE_NAME')
            table_alias = merged_dict.get('CNAME')
            tables[id(subtree)] = (table_name, table_alias, subtree, ptree)
                
        queue += [(c, subtree) for c in reversed(subtree.children)  # type: ignore[misc]
                    if isinstance(c, Tree) and id(c) not in subtrees]
    return tables


def build_children_map(tree):
    """Depth-first iteration.

    Iterates over all the subtrees, never returning to the same node twice (Lark's parse-tree is actually a DAG).
    """
    queue = [tree]
    subtrees = OrderedDict()
    for subtree in queue:
        subtrees[id(subtree)] = subtree
        
        children_map = defaultdict(None)
        for c in subtree.children:
            if c is not None:
                if isinstance(c, Token):
                    children_map[c.value] = c
                else:
                    children_map[c.data] = c
        subtree.children_map = children_map

        # Reason for type ignore https://github.com/python/mypy/issues/10999
        queue += [c for c in reversed(subtree.children)  # type: ignore[misc]
                    if isinstance(c, Tree) and id(c) not in subtrees]

    del queue
    


def main():
    json_grammar = r"""
        ?value: dict
            | list
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null

        list : "[" [value ("," value)*] "]"

        dict : "{" [pair ("," pair)*] "}"
        pair : string ":" value

        string : "\"" /[a-zA-Z0-9 ]\{0,10\}/ "\""

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS

        """
    comp_engine = LarkCompletionEngine(json_grammar, 'value')
    text = '{"8W{0sxM{{}]]vpEC4|i;]V@Jg_#P^j\n?k%noXNt\2#2]a8a\PJru]/`M6gaqb@EhFx"'

    GRAMMAR_PATH = '/workspace/user_code/sql_to_ibis/sql_to_ibis/grammar/sql.lark'
    with open(file=GRAMMAR_PATH) as sql_grammar_file:
        _GRAMMAR_TEXT= sql_grammar_file.read()
    comp_engine = LarkCompletionEngine(_GRAMMAR_TEXT, 'select')
    text = "select column1, cast(column2 as integer) + 1 as my_col2 from my_table"

    valid_regexes = comp_engine.complete(text)
    empty = regex.compile('')
    print(valid_regexes)
    print(valid_regexes == empty)
    print(valid_regexes.fullmatch('"abc', partial=True))
    # end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
    # interactive_parser.parser_state.feed_token(end_token, True)


if __name__ == '__main__':
    main()
