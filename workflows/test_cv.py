import unittest
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError
from cv import (
    JobDescription,
    CritiqueResponse,
    RevisedCoverLetter,
    JobCoverLetterState,
    _job_description_node,
    _cover_letter_node,
    _cover_letter_critique_node,
    _revise_node,
    _should_revise_cover_letter,
    job_description_chain,
    critique_chain,
    revise_chain
)


class TestPydanticModels(unittest.TestCase):
    """Test Pydantic model validation and structure"""
    
    def test_job_description_model_valid(self):
        """Test JobDescription model with valid data"""
        job_desc = JobDescription(extracted_job_description="Software Engineer role")
        self.assertEqual(job_desc.extracted_job_description, "Software Engineer role")
    
    def test_job_description_model_default(self):
        """Test JobDescription model with default value"""
        job_desc = JobDescription(extracted_job_description="")
        self.assertEqual(job_desc.extracted_job_description, "")
    
    def test_critique_response_model_valid(self):
        """Test CritiqueResponse model with valid data"""
        critique = CritiqueResponse(
            critique="Need more specific examples",
            cover_letter="Improved cover letter content"
        )
        self.assertEqual(critique.critique, "Need more specific examples")
        self.assertEqual(critique.cover_letter, "Improved cover letter content")
    
    def test_critique_response_model_no_critique(self):
        """Test CritiqueResponse model with no critique"""
        critique = CritiqueResponse(
            critique=None,
            cover_letter="Good cover letter content"
        )
        self.assertIsNone(critique.critique)
        self.assertEqual(critique.cover_letter, "Good cover letter content")
    
    def test_critique_response_model_missing_cover_letter(self):
        """Test CritiqueResponse model validation fails without cover_letter"""
        with self.assertRaises(ValidationError):
            CritiqueResponse(critique="Some critique")
    
    def test_revised_cover_letter_model_valid(self):
        """Test RevisedCoverLetter model with valid data"""
        revised = RevisedCoverLetter(cover_letter="Revised cover letter content")
        self.assertEqual(revised.cover_letter, "Revised cover letter content")
    
    def test_revised_cover_letter_model_missing_content(self):
        """Test RevisedCoverLetter model validation fails without content"""
        with self.assertRaises(ValidationError):
            RevisedCoverLetter()


class TestNodeFunctions(unittest.TestCase):
    """Test individual node functions"""
    
    @patch('cv.job_description_chain')
    def test_job_description_node(self, mock_chain):
        """Test _job_description_node function"""
        mock_result = Mock()
        mock_result.extracted_job_description = "Test job description"
        mock_chain.invoke.return_value = mock_result
        
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="",
            cover_letter="",
            critique=None
        )
        
        result = _job_description_node(state)
        
        mock_chain.invoke.assert_called_once_with({
            "job_url_content": "<html>job content</html>"
        })
        self.assertEqual(result, {"job_description": "Test job description"})
    
    @patch('cv.create_cover_letter')
    def test_cover_letter_node(self, mock_create_cover_letter):
        """Test _cover_letter_node function"""
        mock_create_cover_letter.return_value = "Generated cover letter"
        
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="",
            critique=None
        )
        
        result = _cover_letter_node(state)
        
        mock_create_cover_letter.assert_called_once()
        self.assertEqual(result, {"cover_letter": "Generated cover letter"})
    
    @patch('cv.critique_chain')
    def test_cover_letter_critique_node(self, mock_chain):
        """Test _cover_letter_critique_node function"""
        mock_result = Mock()
        mock_result.critique = "Needs improvement"
        mock_result.cover_letter = "Updated cover letter"
        mock_chain.invoke.return_value = mock_result
        
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="Initial cover letter",
            critique=None
        )
        
        result = _cover_letter_critique_node(state)
        
        mock_chain.invoke.assert_called_once_with({
            "job_description": "Software Engineer position",
            "resume_str": "test resume",
            "cover_letter": "Initial cover letter"
        })
        self.assertEqual(result, {
            "critique": "Needs improvement",
            "cover_letter": "Updated cover letter"
        })
    
    @patch('cv.revise_chain')
    def test_revise_node(self, mock_chain):
        """Test _revise_node function"""
        mock_result = Mock()
        mock_result.cover_letter = "Revised cover letter"
        mock_chain.invoke.return_value = mock_result
        
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="Initial cover letter",
            critique="Needs specific examples"
        )
        
        result = _revise_node(state)
        
        mock_chain.invoke.assert_called_once_with({
            "job_description": "Software Engineer position",
            "resume_str": "test resume",
            "cover_letter": "Initial cover letter",
            "critique": "Needs specific examples"
        })
        self.assertEqual(result, {"cover_letter": "Revised cover letter"})


class TestConditionalLogic(unittest.TestCase):
    """Test conditional logic functions"""
    
    def test_should_revise_cover_letter_with_critique(self):
        """Test _should_revise_cover_letter returns 'revise' when critique exists"""
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="Initial cover letter",
            critique="Needs improvement"
        )
        
        result = _should_revise_cover_letter(state)
        self.assertEqual(result, "revise")
    
    def test_should_revise_cover_letter_without_critique(self):
        """Test _should_revise_cover_letter returns 'end' when no critique"""
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="Good cover letter",
            critique=None
        )
        
        result = _should_revise_cover_letter(state)
        self.assertEqual(result, "end")
    
    def test_should_revise_cover_letter_with_empty_critique(self):
        """Test _should_revise_cover_letter returns 'revise' when critique is empty string"""
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>job content</html>",
            job_description="Software Engineer position",
            cover_letter="Good cover letter",
            critique=""
        )
        
        result = _should_revise_cover_letter(state)
        self.assertEqual(result, "revise")


class TestMainIntegration(unittest.TestCase):
    """Test main function integration"""
    
    @patch('cv.get_url_content')
    @patch('cv.get_resume_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_function_argument_parsing(self, mock_args, mock_resume, mock_url):
        """Test main function argument parsing"""
        mock_args.return_value = Mock(url="https://example.com/job")
        mock_resume.return_value = "resume data"
        mock_url.return_value = "<html>job content</html>"
        
        with patch('cv.StateGraph') as mock_graph:
            mock_graph_instance = Mock()
            mock_graph.return_value = mock_graph_instance
            mock_graph_instance.add_node.return_value = mock_graph_instance
            mock_graph_instance.add_edge.return_value = mock_graph_instance
            mock_graph_instance.add_conditional_edges.return_value = mock_graph_instance
            mock_compiled_graph = Mock()
            mock_graph_instance.compile.return_value = mock_compiled_graph
            mock_compiled_graph.invoke.return_value = {"cover_letter": "Final cover letter"}
            
            from cv import main
            
            with patch('builtins.print') as mock_print:
                main()
                
                mock_resume.assert_called_once()
                mock_url.assert_called_once_with("https://example.com/job")
                mock_print.assert_called_once_with("Final cover letter")


class TestChainIntegration(unittest.TestCase):
    """Test chain integration without mocking LLM calls"""
    
    def test_job_description_state_type(self):
        """Test JobCoverLetterState type structure"""
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>content</html>",
            job_description="job desc",
            cover_letter="cover letter",
            critique="critique"
        )
        
        self.assertIsInstance(state["resume_str"], str)
        self.assertIsInstance(state["job_url_content"], str)
        self.assertIsInstance(state["job_description"], str)
        self.assertIsInstance(state["cover_letter"], str)
        self.assertIsNotNone(state["critique"])
    
    def test_job_description_state_optional_critique(self):
        """Test JobCoverLetterState with optional critique"""
        state = JobCoverLetterState(
            resume_str="test resume",
            job_url_content="<html>content</html>",
            job_description="job desc",
            cover_letter="cover letter",
            critique=None
        )
        
        self.assertIsNone(state["critique"])