package ida.ipl;
import java.util.List;
import java.util.ArrayList;
public class BoardJobStat {

	int nrOfRequestSent;
	int nrOfRequestProcessed;
	boolean solutionFound;
	int poolSize;
	long slowest;
	long fastest;
	int result;
	List<BoardJob> workList;
	int nrOfJobSent;

	BoardJobStat(List<BoardJob> workList, int poolSize)
	{
		this.nrOfRequestSent = 0;
		this.nrOfRequestProcessed = 0;
		this.solutionFound = false;
		this.poolSize = poolSize;
		this.slowest = -1;
		this.fastest = -1;
		this.result = 0;
		this.workList = new ArrayList<BoardJob>(workList);
	}


	int getWork(List<BoardJob> currentWork, int nrOfJobs, int nrOfWorker)
	{
		if(workList.size() > 0)
		{
			List<BoardJob> removeJobs = new ArrayList<BoardJob>(nrOfJobs);
			int leftJob = 0;	
			int index = 0;
			while(leftJob < nrOfJobs)
			{
				if(index >= workList.size())
				{
					break;
				}
				else
				{
					currentWork.add(workList.get(index));
					removeJobs.add(workList.get(index));
					index = index + nrOfWorker;
					this.nrOfRequestSent++;
				}	
				leftJob++;
			}
			for(BoardJob b : removeJobs)
			{
				workList.remove(b);
			}
			return leftJob;
		}
		return 0;
		
	}

	void updateResult(int result)
	{
		this.result += result;
	}

	int getResult()
	{
		return result;
	}


	void updateTime(long timeTaken)
	{
		if(getRequestProcessed() == 1)
		{
			fastest = timeTaken;
		}
		else if(getRequestProcessed() == poolSize)	
		{
			slowest = timeTaken;
		}
	}

	void printTime()
	{
		if(slowest == -1)
		{
		//	System.out.print("Ongoing Transaction");
		}
		else
		{
		//	System.out.println("time Taken " + (slowest - fastest));
		}
	}

	
	//synchronized 
	void incrementRequestDone()
	{
		nrOfRequestProcessed++;
	}

	//synchronized 
	void incrementRequest()
	{
		nrOfRequestSent++;
	}
	
	//synchronized 
	int getRequestProcessed()
	{
		return nrOfRequestProcessed;
	}

	//synchronized 
	int getRequestSentCount()
	{
		return nrOfRequestSent;
	}

	//synchronized 
	boolean isSolutionFound()
	{
		return solutionFound;
	}

	//synchronized 
	void setSolutionFound()
	{
		solutionFound = true;
	}

	//synchronized 
	boolean print(int bound, boolean startIndex)
	{
		int result = getRequestProcessed();
		if(result == getRequestSentCount() && result == poolSize)
		{
			System.out.print(bound + " ");
			return true;
		}
		else if(startIndex && isSolutionFound())
		{
			System.out.print(bound + " ");
			return true;
			
		}
		/*
		else
		{
			System.out.println("Tried print bound " + result +" ," +  poolSize + "," +  getRequestSentCount());
		}
	
		*/
		return false;
	}
}
