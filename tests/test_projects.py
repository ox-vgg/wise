from pathlib import Path
from src import projects
import pytest


@pytest.fixture(scope="session")
def basedir(tmp_path_factory):
    """
    Creates a temporary directory to act as "Home" directory
    for the Wise Project
    """
    _basedir = tmp_path_factory.mktemp("home")
    _basedir.mkdir(exist_ok=True)

    return _basedir


@pytest.fixture(scope="session")
def patch_home(basedir):
    """
    Injects the temporary directory as root_folder for .wise
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(projects, "WiseTree", projects._WiseTree(basedir))
        yield mp


@pytest.mark.usefixtures("patch_home")
class TestWiseTree:
    def test_wise_folder_created(self, basedir: Path):
        # Base dir must be empty
        assert sum(1 for _ in basedir.iterdir()) == 0

        # This should create the wise_folder on first access
        wise_folder = projects.WiseTree.root
        assert wise_folder.exists()

        # Projects folder should also exist
        projects_folder = projects.WiseTree.projects
        assert projects_folder.exists()

        assert wise_folder.parent == basedir
        assert projects_folder.is_relative_to(wise_folder)

    def test_wise_db_paths(self):
        wise_folder = projects.WiseTree.root
        wise_db = projects.WiseTree.db

        assert wise_db.parent == wise_folder
        assert str(wise_db.absolute()) in projects.WiseTree.dburi


@pytest.mark.usefixtures("patch_home")
class TestWiseProjects:
    def test_create_project(self):

        p = projects.WiseProjectTree("test")
        assert p.location.exists() == False

        # Create the project
        p = projects.WiseProjectTree.create("test")

        # Assert
        assert p.location.is_relative_to(projects.WiseTree.projects)

        assert p.location.exists() == True
        p.delete()
        assert p.location.exists() == False

    def test_create_project_symlink(self, tmp_path):
        # Create the project in destination, with a random file in it
        # to check overwriting
        destination = tmp_path / "elsewhere"
        destination.mkdir()

        p = projects.WiseProjectTree.create("test", destination)

        # Assert
        assert p.location.is_symlink()
        assert p.location.is_relative_to(projects.WiseTree.projects)
        assert p.location.resolve().is_relative_to(destination)

        p.delete()
        assert p.location.exists() == False

    def test_should_raise_exception_if_directory_exists_in_destination(self, tmp_path):
        # Create the project in destination, with a directory in it with same name
        destination = tmp_path / "elsewhere"
        destination.mkdir()

        random_dir = destination / "test"
        random_dir.mkdir()

        # Should raise
        with pytest.raises(ValueError):
            projects.WiseProjectTree.create("test", destination)
