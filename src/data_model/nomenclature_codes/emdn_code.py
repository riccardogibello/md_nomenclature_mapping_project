import re
from typing import Optional, Callable, Any, List

from src.data_model.nomenclature_codes.code import Code
from src.data_model.nomenclature_codes.sentence_data import SentenceData


def find_common_emdn_code(
        code1: str,
        code2: str
) -> tuple[Optional[str], Optional[int]]:
    """
    This function returns the common part of the two EMDN codes.

    :param code1: The first EMDN code.
    :param code2: The second EMDN code.

    :return: The common part of the two EMDN codes.
    """
    common_code = ''
    match_level = 0
    # Get the category of the two codes
    category1 = code1[0]
    category2 = code2[0]

    # If the two codes have the same category
    if category1 == category2:
        common_code = common_code + category1
        match_level = 1

        for i in range(1, min(len(code1), len(code2)), 2):
            # If the two codes have the same two consecutive digits
            if code1[i] == code2[i] and code1[i + 1] == code2[i + 1]:
                match_level = match_level + 1
                common_code = common_code + code1[i] + code1[i + 1]
            else:
                break

        return common_code, match_level
    else:
        return None, match_level


def get_code_level(
        emdn_code_string: str
):
    """
    This function returns the level of the EMDN code.

    :param emdn_code_string: The EMDN code string.

    :return: The level of the EMDN code.
    """
    if emdn_code_string is None or len(emdn_code_string) == 0:
        return -1
    else:
        tot_levels = 1
        # Remove the EMDN category
        emdn_code_string = emdn_code_string[1:]
        # Count the number of levels composed by two digits
        for i in range(0, len(emdn_code_string) - 1, 2):
            tot_levels = tot_levels + 1

        return tot_levels


class EmdnCode(Code):
    """
    Class that stores all the data related to a specific medical device  of the EMDN nomenclature.

    Every src consists of (at least) an alphanumerical identifier (that is the official src) and a synthetic
    description, whose data are stored into data.

    Moreover, it stores the EMDN hierarchy level (level_number) of the current src and whether it is a leaf (is_leaf).
    """

    def __init__(
            self,
            identifier: int,
            emdn_code: str,
            description: str,
            is_leaf: bool,
    ):
        """
        Initializes the class.

        :param identifier: The unique identifier used to identify the code in the database.
        :param emdn_code: The EMDN alphanumeric code used to identify the code in the EMDN nomenclature.
        :param description: The EMDN code description.
        :param is_leaf: Indicates whether the current code is a leaf of the hierarchy.
        """

        super().__init__(
            identifier,
            SentenceData(
                sentence=description,
            )
        )

        self.emdn_code: str = emdn_code

        if type(is_leaf) is not bool:
            raise ValueError("The is_leaf parameter must be a boolean value.")
        self.is_leaf: bool = is_leaf

        self.parent: Optional[EmdnCode | VotedEmdnCode] = None

        self.children: list[EmdnCode | VotedEmdnCode] = []

    def to_json(
            self,
            key__lambda: Optional[dict[str, Callable[[Any], Any]]] = None,
            excluded_keys: Optional[list[str]] = None,
    ) -> dict[str, any]:
        excluded_keys = [
            'parent',
            'children',
            'is_leaf',
        ]

        return super().to_json(
            key__lambda={
                **(key__lambda if key__lambda is not None else {}),
                'emdn_description': lambda description_data: description_data.sentence,
            },
            excluded_keys=excluded_keys,
        )

    def find_descendant_by_id(
            self,
            searched_id: int
    ) -> Optional["EmdnCode"]:
        """
        This method searches the descendant of the current instance with the given identifier.

        :param searched_id: The identifier of the descendant to be searched.

        :return: The descendant with the given identifier, if it is found among the descendants of the current instance.
        None, otherwise.
        """
        if self.identifier == searched_id:
            return self
        else:
            # Search into each child
            for child in self.children:
                found_descendant = child.find_descendant_by_id(
                    searched_id=searched_id
                )
                if found_descendant:
                    return found_descendant

        return None

    def get_descendants(
            self
    ) -> dict[int, "EmdnCode"]:
        """
        This method returns all the descendants of the current instance.

        :return: A dictionary containing all the descendants of the current instance.
        """
        descendants: dict[int, "EmdnCode"] = {}
        for child in self.children:
            descendants[child.identifier] = child
            child_descendants = child.get_descendants()
            descendants.update(child_descendants)

        return descendants

    def find_descendant(
            self,
            searched_code: str
    ) -> Optional["EmdnCode"]:
        """
        This method searches the given code among the current instance's descendants.

        :param searched_code: The EMDN code to be searched among the descendants of the current instance.

        :return: The EMDN code instance corresponding to the searched code, if it is found among the descendants.
        None, otherwise.
        """
        # If the current code is the searched one
        if self.emdn_code == searched_code:
            return self
        else:
            is_descendant = self.should_be_descendant(
                searched_code
            )

            found_descendant: Optional[EmdnCode] = None
            # If the searched EMDN code is a descendant of the current code
            if is_descendant:
                # For each child of the current code
                for child in self.children:
                    # If the current child is the searched one
                    if child.emdn_code == searched_code:
                        # Return it
                        return child
                    # If the searched code is a descendant of the current child
                    elif child.should_be_descendant(
                            child_code=searched_code
                    ):
                        # Search the code among the current child's descendants
                        found_descendant = child.find_descendant(
                            searched_code=searched_code
                        )
                        if found_descendant:
                            break

            return found_descendant

    def should_be_descendant(
            self,
            child_code: str
    ) -> bool:
        """
        This method checks whether the given EMDN code string is a descendant of the current code.

        :param child_code: The EMDN code string to be checked.

        :return: A boolean that indicates whether the given code is a descendant of the current code.
        """
        # Get the EMDN category of the child code and the current code
        child_category = re.sub(
            pattern=r"[0-9]*",
            string=child_code,
            repl=''
        )
        current_category = re.sub(
            pattern=r"[0-9]*",
            string=self.emdn_code,
            repl=''
        )

        # If the categories are different, the child code cannot be a descendant of the current code
        if child_category != current_category or len(child_code) < len(self.emdn_code):
            return False
        else:
            # Get the digits of the child code
            child_code_digits = re.sub(
                pattern=r"[a-zA-Z]*",
                string=child_code,
                repl=''
            )
            # Get the digits of the current code
            current_code_digits = re.sub(
                pattern=r"[a-zA-Z]*",
                string=self.emdn_code,
                repl=''
            )

            # If all the first digits of the current code are shared by the child code, then the
            # child code is a descendant of the current code
            index = 0
            for char in current_code_digits:
                # if any of the digits of the current src is not present in child_digits, then the src is not
                # a child of the current one
                if child_code_digits[index] != char:
                    return False

                index = index + 1

            return True

    def add_to_children(
            self,
            code_to_be_added: "EmdnCode",
            update_vote_count: bool = False,
    ) -> bool:
        """
        This method adds the given code_to_be_added to the children of the current code instance,
        or to the children of one of its descendants.

        :param code_to_be_added: The EMDN code to be added to the nomenclature.
        :param update_vote_count: A boolean that indicates whether the votes of the added code should be updated.

        :return: A boolean that indicates whether the code has been added to the nomenclature.
        """
        # Initialize the boolean that indicates whether the EMDN code
        # is added as child of the current code instance
        is_added = False

        is_added_a_descendant = self.should_be_descendant(
            code_to_be_added.emdn_code
        )
        is_self_a_descendant = code_to_be_added.should_be_descendant(
            self.emdn_code
        )

        # If the currently added code is a descendant of the current instance code
        if is_added_a_descendant:
            # For each child of the current instance
            for child in self.children:
                # If the current child is the added code
                if child.emdn_code == code_to_be_added.emdn_code:
                    # The code has already been added
                    is_added = True
                    # Update the votes of the already present code
                    if update_vote_count:
                        child.add_vote()
                    break
                else:
                    # Try to add the code to its children
                    is_added = child.add_to_children(
                        code_to_be_added=code_to_be_added,
                        update_vote_count=update_vote_count
                    )
                    # As soon as the code is added
                    if is_added:
                        # Break the loop to update the votes of the current code
                        break
            # If the code was not added to the children of the current code
            if not is_added:
                # Set the current instance as the parent of the new code
                code_to_be_added.parent = self
                # Add the new code as the child of the current instance code
                self.children.append(code_to_be_added)
                # Update the votes of the added code, initializing
                # them with the votes of the current instance, plus one
                if update_vote_count:
                    code_to_be_added.add_vote()
                # Update the boolean to indicate that the code has been added
                is_added = True

            # Update the votes of the current code by adding one vote
            if update_vote_count:
                self.add_vote()
        # If the added code is a superior code of the current instance
        elif is_self_a_descendant:
            # Remove the current instance from the children of its parent
            self.parent.children.remove(self)
            # Set the added code as the parent of the current instance
            self.parent = code_to_be_added
            # Add the current instance to the children of the added code
            code_to_be_added.children.append(self)
            # Update the votes of the added code, initializing
            # them with the votes of the current instance, plus one
            if update_vote_count:
                code_to_be_added.add_vote(
                    self.get_votes() + 1
                )
            # Set the boolean so that it indicates that the code has been added
            is_added = True

        return is_added

    def add_vote(
            self,
            votes_to_add: int = 1,
    ):
        pass

    def get_votes(self):
        return 0


class VotedEmdnCode(EmdnCode):

    def __init__(
            self,
            identifier: int,
            emdn_code: str,
            description: str,
            is_leaf: bool,
    ):
        super().__init__(
            identifier,
            emdn_code,
            description,
            is_leaf
        )

        self.vote_count: int = 0

    def add_vote(
            self,
            votes_to_add: int = 1
    ) -> None:
        self.vote_count = self.vote_count + votes_to_add

    def get_votes(self):
        return self.vote_count

    def find_most_voted_children(
            self
    ) -> List["VotedEmdnCode"]:
        best_children: List[VotedEmdnCode] = []
        max_votes = 0

        for child in self.children:
            if child.vote_count > max_votes:
                best_children = [child]
                max_votes = child.vote_count
            elif child.vote_count == max_votes:
                best_children.append(child)

        returned_children = best_children.copy()
        for best_child in best_children:
            returned_children.extend(
                best_child.find_most_voted_children()
            )

        return returned_children
