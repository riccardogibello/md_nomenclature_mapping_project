import re
from typing import Optional, List

from src.data_model.nomenclature_codes.emdn_code import EmdnCode, VotedEmdnCode


class EmdnTreeHandler:
    """
    This is the handler of the EMDN nomenclature tree.
    """

    def __init__(self):
        # The list of EMDN codes of the first level of the tree
        self.code_tree_root: list[EmdnCode | VotedEmdnCode] = []

    def get_codes(
            self
    ) -> tuple[dict[int, EmdnCode], list[int]]:
        """
        Returns all the EMDN codes in the tree.

        :return: A dictionary indexing the EMDN codes by their identifiers and a list of the identifiers.
        """
        returned_codes = {}
        returned_identifiers = []

        # For each EMDN code of the first level
        for root_emdn_code in self.code_tree_root:
            # Add it to the cache
            returned_codes[root_emdn_code.identifier] = root_emdn_code
            # Get all its descendants
            descendants: EmdnCode | dict[int, EmdnCode] = root_emdn_code.get_descendants()
            # Add all the descendants to the cache
            if type(descendants) is dict:
                returned_codes.update(
                    descendants
                )
                returned_identifiers.extend(
                    list(descendants.keys())
                )
            elif type(descendants) is EmdnCode:
                returned_codes[descendants.identifier] = descendants
                returned_identifiers.append(descendants.identifier)

        return returned_codes, returned_identifiers

    def get_code(
            self,
            alphanumeric_code: Optional[str] = None,
            identifier: Optional[int] = None,
    ) -> Optional[EmdnCode]:
        """
        Searches for a specific EMDN code, given its alphanumerical ID.

        :param alphanumeric_code: The alphanumerical ID of the EMDN code that is searched for.
        :param identifier: The identifier of the EMDN code that is searched for.

        :return: It returns the instance corresponding to the EMDN code. If this is not in the cache, returns None.
        """
        # Initialize the searched EMDN code to None
        searched_code: Optional[EmdnCode] = None

        if identifier is not None:
            # Search each root and its descendants for the EMDN code identifier
            for root in self.code_tree_root:
                searched_code = root.find_descendant_by_id(
                    searched_id=identifier
                )
                if searched_code is not None:
                    break
        elif alphanumeric_code is not None:
            # Get the searched EMDN category
            current_code_category = re.sub(
                pattern=r"[0-9]*",
                string=alphanumeric_code,
                repl=''
            )

            # Iterate over the 1st level EMDN codes
            for top_level_code in self.code_tree_root:
                # If the searched and current EMDN codes have the same alphanumerical ID
                if top_level_code.emdn_code == alphanumeric_code:
                    # Return the current code
                    return top_level_code
                else:
                    # Get the current EMDN category
                    top_level_code_category = re.sub(
                        pattern=r"[0-9]*",
                        string=top_level_code.emdn_code,
                        repl=''
                    )
                    # If the searched and current EMDN codes have the same category
                    if top_level_code_category == current_code_category:
                        # For each child of the current EMDN code
                        for child in top_level_code.children:
                            # Search for the EMDN code among the descendants of the current EMDN code
                            searched_code = child.find_descendant(
                                searched_code=alphanumeric_code
                            )
                            # Break as soon as the EMDN code is found
                            if searched_code is not None:
                                break
        else:
            raise ValueError("Either the alphanumerical code or the identifier of the EMDN code should be given.")

        return searched_code

    def add_to_tree(
            self,
            added_code: EmdnCode | VotedEmdnCode,
            update_vote_count: bool = False,
    ) -> None:
        """
        Adds a new EMDN code to the tree.

        :param added_code: The EMDN code to be added to the tree.
        :param update_vote_count: A boolean indicating whether the vote count of the code should be updated.
        """
        # If the current tree root is empty
        if len(self.code_tree_root) == 0:
            # Add the new EMDN code to the roots
            self.code_tree_root.append(
                added_code
            )
            # Add its vote if the code is a VotedEmdnCode
            if update_vote_count:
                added_code.add_vote()
        else:
            # Keep track whether the code is added to the tree
            added = False
            # For each code among the roots
            for root_code in self.code_tree_root:
                # If the root corresponds to the added code
                if root_code.emdn_code == added_code.emdn_code:
                    # Set the value indicating that the code is added to the tree
                    added = True
                    # Update the vote of the code in the tree, if it is a VotedEmdnCode
                    if update_vote_count:
                        root_code.add_vote()
                    break
                else:
                    # Check if the new code is a super-code of the current root
                    is_root_code_a_descendant = added_code.should_be_descendant(
                        root_code.emdn_code
                    )
                    # If the new code is a super-code of the current root
                    if is_root_code_a_descendant:
                        # Set the new code as the parent of the current root
                        root_code.parent = added_code
                        # Add the root code to the children of the new code
                        added_code.children.append(root_code)
                        # Add the new code to the roots
                        self.code_tree_root.remove(root_code)
                        self.code_tree_root.append(
                            added_code
                        )
                        # Keep track that the code is added to the tree
                        added = True
                        # Update the vote, if the code is a VotedEmdnCode, with
                        # the descendant votes, plus one
                        if update_vote_count:
                            added_code.add_vote(
                                root_code.get_votes() + 1
                            )
                        break
                    else:
                        # Try to add the new code to the children of the current root
                        added = root_code.add_to_children(
                            added_code,
                            update_vote_count
                        )
                        # If the code was added to the children
                        if added:
                            # Break the loop
                            break

            # If the code was not added to any of the roots, it is a root itself
            if not added:
                # Add it to the roots anyway
                self.code_tree_root.append(
                    added_code
                )
                if update_vote_count:
                    # Add its vote if the code is a VotedEmdnCode
                    added_code.add_vote()

    def get_most_voted_codes(
            self
    ) -> List[VotedEmdnCode]:
        """
        Returns the most voted EMDN codes in the tree.

        :return: A list of the most voted EMDN codes in the tree.
        """
        # Initialize the list of most voted codes
        most_voted_codes = []

        count__root_codes: dict[int, list[VotedEmdnCode]] = {}
        # For each root code
        for root_code in self.code_tree_root:
            # Get the votes of the current code
            current_votes = root_code.get_votes()
            if count__root_codes.get(current_votes) is None:
                modified_list: list[VotedEmdnCode] = []
                count__root_codes[current_votes] = modified_list
            else:
                modified_list = count__root_codes[current_votes]
            modified_list.append(root_code)

        # Sort the votes in descending order
        sorted_votes = sorted(
            count__root_codes.keys(),
            reverse=True
        )
        added_roots = 0
        for sorted_vote in sorted_votes:
            # Get the codes with the current number of votes
            current_root_voted_codes = count__root_codes[sorted_vote]
            added_roots += len(current_root_voted_codes)
            # Add the codes to the list of most voted codes
            most_voted_codes.extend(
                current_root_voted_codes
            )
            # For each root,
            for added_root in current_root_voted_codes:
                # Find all the most voted descendants, up to the fourth level
                most_voted_codes.extend(
                    added_root.find_most_voted_children()
                )

            # If the current roots added is already at least two
            if added_roots >= 2:
                # Do not add any other option
                break

        return most_voted_codes
