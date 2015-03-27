package ida.ipl;

import ibis.ipl.Ibis;
import ibis.ipl.IbisCapabilities;
import ibis.ipl.IbisCreationFailedException;
import ibis.ipl.IbisFactory;
import ibis.ipl.IbisIdentifier;
import ibis.ipl.MessageUpcall;
import ibis.ipl.PortType;
import ibis.ipl.ReadMessage;
import ibis.ipl.ReceivePort;
import ibis.ipl.ReceivePortIdentifier;
import ibis.ipl.ReceiveTimedOutException;
import ibis.ipl.Registry;
import ibis.ipl.SendPort;
import ibis.ipl.WriteMessage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

public class Ida
  implements MessageUpcall
{
	public static final boolean PRINT_SOLUTION = false;
	private Ibis ibis;
	IbisCapabilities ibisCap;
	Registry registry;
	PortType oneToOneUpcallPort;
	PortType oneToOneExplicitPort;
	PortType manyToOneUpcallPort;
	PortType manyToOneExplicitPort;
	PortType manyToManyUpcallPort;
	IbisIdentifier masterId;
	private static int finalResult = 0;
	private static int finalBound = 0;
	private static BoardCache[] thread_cache;
	private static BoardCache IplCache;
	private String fileName;
	private int length;
	private int depth;
	private int nrOfJobs;
	private static int workDone = 0;

	public boolean done = false;
	public long fastest;
	public long slowest;
	HashMap<Integer, BoardJobStat> jobStat;
	SortedSet<Integer> boardSet;
	int maxBound;
	List<Long> computationTime, communicationTime;
	List<Integer> boardGenerated;

		public void upcall(ReadMessage message)
		throws IOException
		{
			byte b = message.readByte();
			byte bound = message.readByte();
			boolean resultFlag = false;

			if (this.masterId.equals(this.ibis.identifier()))
			{
				byte[] longBytes = new byte[8];
				message.readArray(longBytes);
				message.finish();

				ByteArrayInputStream bais = null;
				DataInputStream dis = null;
				long timeTaken = 0L;
				try {
					bais = new ByteArrayInputStream(longBytes);
					dis = new DataInputStream(bais);
					timeTaken = dis.readLong();
				}
				catch (Exception e)
				{
					e.printStackTrace();
				}
				finally
				{
					try {
						if (dis != null)
						{
							dis.close();
						}
						if (bais != null)
						{
							bais.close();
						}
					}
					catch (Exception e)
					{
						e.printStackTrace();
					}
				}
				setMaxBound(bound);

				if (b != 0)
					setSolutionFound(bound, b);
				resultFlag = updateAll(bound, timeTaken);
			}

			if (b != 0)
			{
				if (this.masterId.equals(this.ibis.identifier()))
				{
					if (resultFlag)
					{
						this.done = true;
						//System.out.println("WorkDone in Master" + workDone);
					}

				}
				else if (!this.masterId.equals(this.ibis.identifier()))
				{
					this.done = true;
				}

			}

			if (b != 0)
			{
				if (!this.masterId.equals(this.ibis.identifier()))
				{
					//System.out.println("Worker " + this.ibis.identifier() + "=" + workDone);
					Collections.sort(communicationTime);
					Collections.sort(computationTime);
					Collections.sort(boardGenerated);

					//System.out.println("median Communication TIme" + communicationTime.get(communicationTime.size() / 2));
					//System.out.println("median Computation TIme" + computationTime.get(computationTime.size() / 2));
					//System.out.println("BoardGenerated" + boardGenerated.get(boardGenerated.size() / 2));
					this.ibis.end();
					System.exit(0);
				}
			}
		}

	synchronized boolean updateAll(int bound, long timeTaken)
	{
		incrementRequestProcessedCount(bound);
		updateTime(bound, timeTaken);
		boolean resultFlag = printStat(bound);

		if (resultFlag)
		{
			this.done = true;
			//System.out.println("WorkDone in Master" + workDone);
		}
		return resultFlag;
	}

	void setMaxBound(int bound)
	{
		if (this.maxBound < bound)
		{
			this.maxBound = bound;
		}
	}

	void incrementRequestProcessedCount(int bound)
	{
		BoardJobStat bjs = (BoardJobStat)this.jobStat.get(Integer.valueOf(bound));

		bjs.incrementRequestDone();
	}

	void setSolutionFound(int bound, int nrOfResult)
	{
		BoardJobStat bjs = (BoardJobStat)this.jobStat.get(Integer.valueOf(bound));
		//System.out.println("Set processed Sent for " + bound);
		bjs.setSolutionFound();
		bjs.updateResult(nrOfResult);
	}

	public Ida(int depth, int nrOfJobs)
	{
		this.fastest = 0L;
		this.slowest = 0L;
		this.nrOfJobs = nrOfJobs;
		this.jobStat = new HashMap();
		this.maxBound = 0;
		this.depth = depth;
		this.computationTime = new ArrayList<Long>();
		this.communicationTime = new ArrayList<Long>();
		this.boardGenerated =  new ArrayList<Integer>();

		this.boardSet = new TreeSet();

		this.ibisCap = new IbisCapabilities(new String[] { "elections.strict", "membership.totally.ordered", "termination", "closed.world" });

		this.oneToOneExplicitPort = new PortType(new String[] { "communication.reliable", "serialization.object", "receive.timeout", "receive.explicit", "connection.onetoone" });

		this.oneToOneUpcallPort = new PortType(new String[] { "communication.reliable", "serialization.object", "serialization.data", "receive.autoupcalls", "connection.onetoone" });

		this.manyToOneExplicitPort = new PortType(new String[] { "communication.reliable", "serialization.object", "receive.timeout", "receive.explicit", "connection.manytoone", "connection.downcalls" });

		this.manyToOneUpcallPort = new PortType(new String[] { "communication.reliable", "serialization.object", "serialization.data", "receive.autoupcalls", "connection.manytoone" });

		this.manyToManyUpcallPort = new PortType(new String[] { "communication.reliable", "serialization.data", "receive.autoupcalls", "connection.manytomany", "connection.downcalls" });
	}

	private void updateTime(int bound, long timeTaken)
	{
		BoardJobStat bjs = (BoardJobStat)this.jobStat.get(Integer.valueOf(bound));
		bjs.updateTime(timeTaken);
	}

	private int masterWork(Board board, BoardCache cache, ArrayList<BoardJob> boardJobList)
	{
		List newBoundList = new ArrayList();

		long start = System.currentTimeMillis();

		int bound = board.bound();
		if (board.distance() == 0)
		{
			return 1;
		}

		if (board.distance() > board.bound())
		{
			return 0;
		}

		boardJobList.add(new BoardJob(board, bound));
		int j = 0;

		while (j < this.depth)
		{
			newBoundList.clear();
			while (boardJobList.size() != 0)
			{
				BoardJob tempJob = (BoardJob)boardJobList.remove(0);

				Board tempBoard = tempJob.getBoard();
				Board[] children;
				if (cache != null) {
					children = tempBoard.makeMoves(cache);
				}
				else {
					children = tempBoard.makeMoves();
				}

				int i = 0;

				for (i = 0; i < children.length; i++)
				{
					if (children[i] != null)
					{
						if (children[i].distance() == 0)
						{
							return 1;
						}

						newBoundList.add(new BoardJob(children[i], bound));
					}
				}

				if (boardJobList.size() == 0)
				{
					boardJobList.addAll(newBoundList);
					break;
				}
			}
			j++;
		}

		workDone += boardJobList.size();
		Collections.sort(boardJobList);

		long end = System.currentTimeMillis();
		//System.out.println("Master Time :" + (end - start));
		//System.out.println("Master work for bound" + bound + " =" + boardJobList.size());

		return 0;
	}

	void insertWorkCount(int bound, List<BoardJob> insertWork)
	{
		if (this.jobStat.containsKey(Integer.valueOf(bound)))
		{
			return;
		}
		else
		{
			BoardJobStat bjs = new BoardJobStat(insertWork, insertWork.size());
			this.jobStat.put(Integer.valueOf(bound), bjs);
		}
		this.boardSet.add(Integer.valueOf(bound));
	}

	boolean printStat(int bound)
	{
		if (this.boardSet.size() == 0)
			return false;
		SortedSet subSet = this.boardSet.tailSet(this.boardSet.first());
		Iterator it = subSet.iterator();
		Set removeSet = new HashSet();
		boolean resultFlag = false;
		int i = 0;
		while (it.hasNext())
		{
			Integer boundVal = (Integer)it.next();

			BoardJobStat bjs = (BoardJobStat)this.jobStat.get(boundVal);

			boolean resultVal = false;
			if(i == 0)
			{
				resultVal = bjs.print(boundVal, true);
				i++;
			}
			else
				resultVal = bjs.print(boundVal, false);
			if (!resultVal)
				break;
			removeSet.add(boundVal);
			if (bjs.isSolutionFound())
			{
		//		System.out.println("Result Set");
				resultFlag = true;
				finalBound = boundVal.intValue();
				finalResult = bjs.getResult();
				break;
			}

		}

		it = removeSet.iterator();
		while (it.hasNext())
		{
			this.boardSet.remove(it.next());
		}

		if (resultFlag) {
			return true;
		}
		return false;
	}

	private void master(Board board, boolean usecache)
		throws Exception
		{
			BoardCache cache;
			if (usecache)
				cache = new BoardCache();
			else {
				cache = null;
			}
			ReceivePort receiver = this.ibis.createReceivePort(this.manyToOneExplicitPort, "master");
			receiver.enableConnections();

			ReceivePort receiveUpcall = this.ibis.createReceivePort(this.manyToManyUpcallPort, "upcall", this);
			receiveUpcall.enableConnections();
			receiveUpcall.enableMessageUpcalls();

			int bound = 0;
			this.ibis.registry().waitUntilPoolClosed();

			System.out.println("Running IDA*, initial board:");
			System.out.println(board);

			int poolSize = this.ibis.registry().getPoolSize();

			if (1 == poolSize)
			{
				long start = System.currentTimeMillis();
				solve(board, usecache);
				long end = System.currentTimeMillis();

				System.err.println("ida took " + (end - start) + " milliseconds");
				return;
			}

			long start = System.currentTimeMillis();

			ArrayList completeboardJobList = new ArrayList();
			ArrayList newList = new ArrayList();
			bound = board.distance();
			System.out.print("Try bound ");

			board.setBound(bound);

			int iListIndex = 0;

			boolean generateWork = true;
			int currBound = bound;
			int generatedBound = bound;
			int sentBound = bound;
			long communicationTime = 0L;
			while (!this.done)
			{
				if (generateWork)
				{
					if (this.done) {
						break;
					}

					int result = masterWork(board, cache, newList);
					if (result == 1) {
						break;
					}
					
					insertWorkCount(bound, newList);	
					generatedBound = bound;
					bound += 2;
					board.setBound(bound);
					newList.clear();
					generateWork = false;
				}

				long startc = System.currentTimeMillis();

				ReadMessage reply = null;
				boolean whileEscape = false;
				while (!whileEscape) {
					try
					{
						reply = receiver.receive(100L);
						whileEscape = true;
					}
					catch (ReceiveTimedOutException rtoe)
					{
						if (this.done)
							whileEscape = true;
					}
				}
				if (this.done)
					break;
				ReceivePortIdentifier requestor = (ReceivePortIdentifier)reply.readObject();

				reply.finish();

				SendPort replyPort = this.ibis.createSendPort(this.oneToOneExplicitPort);
				replyPort.connect(requestor);

				WriteMessage data = replyPort.newMessage();

				/*
				BoardJob currentWork = (BoardJob)completeboardJobList.remove(0);
				Board currentBoard = currentWork.getBoard();
				int boundVal = currentWork.getBound();
				*/
				List<BoardJob> sendWork = new ArrayList<BoardJob>(nrOfJobs);
				int totalJobs = 0;
				while(totalJobs < nrOfJobs && sentBound < generatedBound)	
				{
					BoardJobStat bjs = jobStat.get(sentBound);
					int returnJobs = bjs.getWork(sendWork, nrOfJobs, poolSize - 1);
					if(returnJobs == 0)
					{
						sentBound += 2;
						if(sentBound == generatedBound)
						{
							int result = masterWork(board, cache, newList);
							if (result == 1) {
								break;
							}

							insertWorkCount(bound, newList);	
							generatedBound = bound;
							bound += 2;
							board.setBound(bound);
							newList.clear();
							generateWork = false;
						}
					}
					totalJobs += returnJobs;
				}
				ListBoardJob  lbj = new ListBoardJob(sendWork);
				
				data.writeObject(lbj);
				//data.writeInt(boundVal);
				data.finish();
				replyPort.close();
				long endc = System.currentTimeMillis();
				//insertOrIncrmentSendCount(boundVal, ((Integer)boardMap.get(Integer.valueOf(boundVal))).intValue());

				if(generatedBound - sentBound < 4)
				{
					generateWork = true;
				}
				communicationTime += endc - startc;
			}

		//	System.out.println("Communication Time" + communicationTime);

			long end = System.currentTimeMillis();

			IbisIdentifier[] joinedIbises = this.ibis.registry().joinedIbises();

			for (IbisIdentifier joinedIbis : joinedIbises)
			{
				if (!joinedIbis.equals(this.masterId))
				{
					SendPort sendResultToManyUpcall = this.ibis.createSendPort(this.manyToManyUpcallPort);
					sendResultToManyUpcall.connect(joinedIbis, "upcall");

					WriteMessage messageResult = sendResultToManyUpcall.newMessage();

					messageResult.writeByte((byte)20);
					messageResult.writeByte((byte)0);
					messageResult.finish();
					sendResultToManyUpcall.close();
				}
			}

			System.out.println();
			System.out.println("result is " + finalResult + " solutions of " + finalBound + " steps");

			System.err.println("ida took " + (end - start) + " milliseconds");
		}

	private void worker(boolean cacheFlag)
		throws Exception
		{
			BoardCache cache = null;

			SendPort sendPort = this.ibis.createSendPort(this.manyToOneExplicitPort);
			sendPort.connect(this.masterId, "master");

			ReceivePort receivePort = this.ibis.createReceivePort(this.oneToOneExplicitPort, "some");
			receivePort.enableConnections();

			ReceivePort receiveUpcall = this.ibis.createReceivePort(this.manyToManyUpcallPort, "upcall", this);
			receiveUpcall.enableConnections();
			receiveUpcall.enableMessageUpcalls();

			while (!this.done)
			{
				long startCommunication = System.currentTimeMillis();
				WriteMessage request = sendPort.newMessage();
				request.writeObject(receivePort.identifier());
				request.finish();

				boolean whileEscape = false;

				ReadMessage reply = null;
				while (!whileEscape) {
					try
					{
						reply = receivePort.receive(100L);
						whileEscape = true;
					}
					catch (ReceiveTimedOutException rtoe)
					{
						if (this.done) {
							whileEscape = true;
						}
					}
				}
				ListBoardJob boardList = (ListBoardJob)reply.readObject();
				//int boundVal = reply.readInt();

				reply.finish();
				long endCommunication = System.currentTimeMillis();
				communicationTime.add((endCommunication - startCommunication));
				BoardJob currentWork = boardList.getNextBoardJob();
				while(currentWork != null)
				{
					int boundVal = currentWork.getBound();

					long startComputation = System.currentTimeMillis();
					int result = 0;
					Board board = currentWork.getBoard();
					int startWD = workDone;
					if (cacheFlag)
					{
						cache = new BoardCache();
						result = solutions(board, cache);
					}
					else {
						result = solutions(board);
					}
					long endComputation = System.currentTimeMillis();
					long tim = endComputation - startComputation;
					computationTime.add(tim);

					byte[] longBytes = null;
					ByteArrayOutputStream baos = null;
					DataOutputStream dos = null;
					try {
						baos = new ByteArrayOutputStream();
						dos = new DataOutputStream(baos);
						dos.writeLong(tim);
						longBytes = baos.toByteArray();
					}
					catch (IOException ioe)
					{
						ioe.printStackTrace();
					}
					finally {
						try {
							if (dos != null)
							{
								dos.close();
							}
							if (baos != null)
							{
								baos.close();
							}

						}
						catch (Exception e)
						{
							e.printStackTrace();
						}

					}

					SendPort sendResultUpcall = this.ibis.createSendPort(this.manyToManyUpcallPort);
					sendResultUpcall.connect(this.masterId, "upcall");

					WriteMessage messageResult = sendResultUpcall.newMessage();
					messageResult.writeByte((byte)result);
					messageResult.writeByte((byte)boundVal);

					messageResult.writeArray(longBytes);
					messageResult.finish();

					sendResultUpcall.close();
					currentWork = boardList.getNextBoardJob();
				}
			}
		}

	public void init(int threads, int length)
	{
		this.length = length;
	}

	public void run(Board board, boolean cache)
		throws Exception
		{
			try
			{
				this.ibis = IbisFactory.createIbis(this.ibisCap, null, new PortType[] { this.oneToOneExplicitPort, this.manyToOneExplicitPort, this.manyToManyUpcallPort });
				this.registry = this.ibis.registry();

				this.masterId = this.registry.elect("master");

				boolean master = this.masterId.equals(this.ibis.identifier());

				if (master)
				{
					master(board, cache);
				}
				else
				{
					worker(cache);
				}

				this.ibis.end();
			}
			catch (IbisCreationFailedException ie)
			{
				ie.printStackTrace();
			}
			catch (IOException ie)
			{
				ie.printStackTrace();
			}
		}

	static int solutions(Board board, BoardCache cache)
	{
		if (board.distance() == 0) {
			return 1;
		}

		if (board.distance() > board.bound()) {
			return 0;
		}

		Board[] children = board.makeMoves(cache);
		int result = 0;

		for (int i = 0; i < children.length; i++) {
			if (children[i] != null) {
				result += solutions(children[i], cache);
				workDone += 1;
			}
		}
		cache.put(children);
		return result;
	}

	static int solutions(Board board)
	{
		if (board.distance() == 0) {
			return 1;
		}

		if (board.distance() > board.bound()) {
			return 0;
		}

		Board[] children = board.makeMoves();
		int result = 0;

		for (int i = 0; i < children.length; i++) {
			if (children[i] != null) {
				result += solutions(children[i]);
				workDone += 1;
			}
		}
		return result;
	}

	private static void solve(Board board, boolean useCache)
	{
		BoardCache cache = null;
		if (useCache) {
			cache = new BoardCache();
		}
		int bound = board.distance();

		System.out.print("Try bound ");
		int solutions;
		do { board.setBound(bound);

			System.out.print(bound + " ");
		//	System.out.flush();
			if (useCache)
				solutions = solutions(board, cache);
			else {
				solutions = solutions(board);
			}

			bound += 2; }
		while (solutions == 0);

		System.out.println("\nresult is " + solutions + " solutions of " + board.bound() + " steps");
	}

	public static void main(String[] args)
	{
		String fileName = null;
		boolean cache = true;

		int threads = 1;

		int length = 103;

		int depth = 1;
		
		int nrOfJobs = 1;

		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("--file")) {
				fileName = args[(++i)];
			} else if (args[i].equals("--nocache")) {
				cache = false;
			} else if (args[i].equals("--threads")) {
				i++;
				threads = Integer.parseInt(args[i]);
			} else if (args[i].equals("--length")) {
				i++;
				length = Integer.parseInt(args[i]);
			} else if (args[i].equals("--depth")) {
				i++;
				depth = Integer.parseInt(args[i]);
			} else if (args[i].equals("--jobs")) {
				i++;
				nrOfJobs = Integer.parseInt(args[i]);
			}else {
				System.err.println("No such option: " + args[i]);
				System.exit(1);
			}

		}

		Board initialBoard = null;

		if (fileName == null)
			initialBoard = new Board(length);
		else {
			try {
				initialBoard = new Board(fileName);
			} catch (Exception e) {
				System.err.println("could not initialize board from file: " + e);

				System.exit(1);
			}

		}

		try
		{
			Ida m = new Ida(depth, nrOfJobs);
			m.init(threads, length);
			m.run(initialBoard, cache);
		}
		catch (Exception e)
		{
			e.printStackTrace(System.err);
		}
	}
}
