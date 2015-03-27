package ida.ipl;

import java.lang.Comparable;
import java.io.Serializable;

class BoardJob implements Comparable<BoardJob>, Serializable{
	private static final long serialVersionUID = -960661472047747221L;
	Board board;
	int bound;

	BoardJob(Board board, int bound)
	{
		this.board = board;
		this.bound = bound;
	}

	Board getBoard()
	{
		return board;
	}

	int getBound()
	{
		return bound;
	}

	public int compareTo(BoardJob comparedObject)
	{
		if(this.board.distance() < comparedObject.getBoard().distance())
		{
			return -1;
		}
		else if(this.board.distance() == comparedObject.getBoard().distance())
			return 0;
		else	
			return 1;
	}
}

