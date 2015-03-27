package ida.ipl;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

class ListBoardJob implements Serializable{
	private static final long serialVersionUID = 6398838388956030131L;	
	List<BoardJob> inputBoards;
	private int currentWork;

	ListBoardJob(List<BoardJob> inputList)
	{
		inputBoards = new ArrayList<BoardJob>(inputList);
		currentWork = 0;
	}
	
	BoardJob getNextBoardJob()
	{
		if(currentWork == inputBoards.size())
		{
			return null;
		}
		else
		{
			BoardJob returnBoard = inputBoards.get(currentWork);
			currentWork++;
			return returnBoard;
		}
	}
}
