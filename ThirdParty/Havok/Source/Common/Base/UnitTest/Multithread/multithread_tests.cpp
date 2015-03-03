/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>
#include <Common/Base/Thread/Thread/hkThread.h>

#define N_THREADS 8
#define N_STEPS 10
#define N_ENTITIES 80

#include <Common/Base/KeyCode.h>
class hkThread;

class FakeClass : public hkReferencedObject
{
public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_DEMO);

	FakeClass() : m_value( 0 ){}
	~FakeClass(){}

	void markForRead()
	{
		m_multiThreadCheck.markForRead();
	}
	void markForWrite()
	{
		m_multiThreadCheck.markForWrite();
	}
	void unmarkForRead()
	{
		m_multiThreadCheck.unmarkForRead();
	}
	void unmarkForWrite()
	{
		m_multiThreadCheck.unmarkForWrite();
	}

	class hkMultiThreadCheck m_multiThreadCheck;
	hkReal m_value;

};

// Fake job queue to exercise multithreading atomics
struct	FakeJobQueue
{
private:
	// Abstract struct to add to job queue
	struct IJob
	{
		virtual ~IJob() {}
		virtual void run() const = 0;
	};

	template < typename T >
	struct Box : public IJob
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, Box );

		inline Box( const T& job ) : m_job( job ) {}
		inline void run() const { m_job.run(); }

		mutable T m_job;
	};

public:
	FakeJobQueue( int numThreads = -1 )
	{
		m_numPendings	=	0;
		m_jobsLock		=	new hkCriticalSection( 0 );
		m_newJobEvent	=	new hkSemaphore(0, 10000);
		m_endJobEvent	=	new hkSemaphore(0, 10000);
		m_endThreadEvent=	new hkSemaphore(0, 10000);

		for( int i = 0; i < N_THREADS; ++i )
		{
			m_threads[i] = new hkThread();
			m_threads[i]->startThread( &threadStart, this, "" );
		}	
	}

	~FakeJobQueue()
	{
		m_jobsLock->enter();
		for ( int i = 0; i < N_THREADS; ++i )
		{
			m_jobs.pushBack( ( IJob* )1 );
		}
		m_jobsLock->leave();

		m_newJobEvent->release( N_THREADS );		
		waitForCompletion();

		for( int i = 0; i < N_THREADS; ++i )
		{
			m_endThreadEvent->acquire();
		}

		for ( int i = 0; i < N_THREADS; ++i )
		{
			delete m_threads[i];
		}
		delete m_newJobEvent;
		delete m_endJobEvent;
		delete m_endThreadEvent;
		delete m_jobsLock;	
	}

	void waitForCompletion()
	{
		bool wait;
		do
		{
			wait = false;
			m_jobsLock->enter();
			wait = m_numPendings || m_jobs.getSize();
			m_jobsLock->leave();

			if( wait )
			{
				m_endJobEvent->acquire();
			}
		}
		while ( wait );
	}

	template < typename T >
	void appendJob( const T& job )
	{
		push( new Box< T >( job ) );
	}

private:
	void push( IJob* job )
	{
		m_jobsLock->enter();
		m_jobs.pushBack( job );		
		m_jobsLock->leave();

		m_newJobEvent->release();
	}

	static void* HK_CALL threadStart( void* arg )
	{
		hkMemoryRouter memoryRouter;
		hkMemorySystem::getInstance().threadInit( memoryRouter, "multithread_tests" );
		hkBaseSystem::initThread( &memoryRouter );
	
		( ( FakeJobQueue* )arg )->threadMain();

		hkBaseSystem::quitThread();
		hkMemorySystem::getInstance().threadQuit( memoryRouter );
		return 0;
	}	

	// Main execution loop per thread in the job queue. Executes jobs until a 'finish' job is encountered
	void threadMain()
	{
		while ( true )
		{
			m_newJobEvent->acquire();
			IJob* ijob = 0;

			m_jobsLock->enter();
			if( m_jobs.getSize() )
			{
				ijob = m_jobs.back();
				m_numPendings++;
				m_jobs.popBack();
				if ( m_jobs.getSize() )
				{
					m_newJobEvent->release();
				}
			}
			m_jobsLock->leave();

			if ( ijob )
			{				
				const bool finalJob = ( ijob == ( void* )1 );
				if ( !finalJob )
				{
					ijob->run();
					delete ijob;
				}
				m_jobsLock->enter();
				m_numPendings--;
				if ( m_jobs.getSize() )
				{
					m_newJobEvent->release();
				}
				m_jobsLock->leave();

				m_endJobEvent->release();
				if ( finalJob )
				{
					break;
				}
			}
		}
		m_endJobEvent->release();
		m_endThreadEvent->release();
	}

private:
	hkThread* 					m_threads[N_THREADS];
	hkCriticalSection*			m_jobsLock;	
	hkSemaphore*				m_newJobEvent;
	hkSemaphore*				m_endJobEvent;
	hkSemaphore*				m_endThreadEvent;
	hkArray< IJob* >			m_jobs;
	int							m_numPendings;
};

// Fake job: exercises various multithreading features
struct ReadsAndWritesJob
{
	ReadsAndWritesJob( const ReadsAndWritesJob& job )
	{
		m_itemsToRead = job.m_itemsToRead;
		m_itemsToUpdate = job.m_itemsToUpdate;
		m_itemsLock = job.m_itemsLock;
	}

	ReadsAndWritesJob( hkArray< FakeClass* >* itemsToRead, hkArray< FakeClass* >& itemsToUpdate, hkCriticalSection* itemsLock )
	{
		m_itemsToRead = itemsToRead;
		m_itemsToUpdate = itemsToUpdate;
		m_itemsLock = itemsLock;
	}

	// Exercise markForRead and markForWrite on a set of objects and carry out various memory allocations
	void run() const
	{
		for ( int i = 0; i < m_itemsToUpdate.getSize(); ++i )
		{
			int solverBufferSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, 2000);
			// Solver buffer is only available when physics is enabled
#if defined(HK_FEATURE_PRODUCT_PHYSICS_2012)
			char* solverBuffer = hkMemSolverBufAlloc< char >( solverBufferSize );
#else
			char* solverBuffer = hkMemHeapBufAlloc< char >( solverBufferSize );
#endif

			hkCriticalSectionLock lock( m_itemsLock );

			for ( int j = 800; j < 1000; ++j )
			{
				char* buffer1 = hkAllocateChunk<char>( j, HK_MEMORY_CLASS_DEMO );
				char* buffer2 = hkAllocateChunk<char>( j, HK_MEMORY_CLASS_DEMO );

				hkDeallocateChunk<char>( buffer1, j, HK_MEMORY_CLASS_DEMO );
				hkDeallocateChunk<char>( buffer2, j, HK_MEMORY_CLASS_DEMO );
			}

			FakeClass* buffer = hkAllocateStack< FakeClass >( (*m_itemsToRead).getSize() );

			for ( int j = 0; j < (*m_itemsToRead).getSize(); ++j )
			{
				if ( j != i )
				{
					(*m_itemsToRead)[j]->markForRead();

					buffer[j].m_value = (*m_itemsToRead)[j]->m_value;

					char* tempBuffer = hkAllocateStack< char >( 1000 );

					hkDeallocateStack< char >( tempBuffer, 1000 );

					(*m_itemsToRead)[j]->unmarkForRead();
				}
			}

			m_itemsToUpdate[i]->markForWrite();

			for ( int j = 900; j < 1100; ++j )
			{
				char* buffer1 = hkAllocateChunk<char>( j, HK_MEMORY_CLASS_DEMO );
				char* buffer2 = hkAllocateChunk<char>( j, HK_MEMORY_CLASS_DEMO );

				hkDeallocateChunk<char>( buffer1, j, HK_MEMORY_CLASS_DEMO );
				hkDeallocateChunk<char>( buffer2, j, HK_MEMORY_CLASS_DEMO );
			}

			hkReal val = 0;
			for ( int j = 0; j < (*m_itemsToRead).getSize(); ++j )
			{
				val += buffer[j].m_value;
			}
			m_itemsToUpdate[i]->m_value = val / m_itemsToUpdate[i]->m_value;

			hkDeallocateStack< FakeClass >( buffer, (*m_itemsToRead).getSize() );

			m_itemsToUpdate[i]->unmarkForWrite();

#if defined(HK_FEATURE_PRODUCT_PHYSICS_2012)
			hkMemSolverBufFree( solverBuffer, solverBufferSize );
#else
			hkMemHeapBufFree( solverBuffer, solverBufferSize );
#endif
		}
	}

private:
	hkArray< FakeClass* >* m_itemsToRead;
	hkArray< FakeClass* > m_itemsToUpdate;
	hkCriticalSection* m_itemsLock;
};

// Fake job: carry out lots of reads
struct LotsOfReadsJob
{
	LotsOfReadsJob( const LotsOfReadsJob& job )
	{
		m_itemsToRead = job.m_itemsToRead;
	}

	LotsOfReadsJob( hkArray< FakeClass* >* itemsToRead )
	{
		m_itemsToRead = itemsToRead;
	}

	// Call markForRead a lot 
	void run() const
	{
		int sz = (*m_itemsToRead).getSize();
		for ( int i = 0; i < sz; ++i )
		{
			(*m_itemsToRead)[i]->markForRead();
			(*m_itemsToRead)[i]->unmarkForRead();
		}
	}

private:
	hkArray< FakeClass* >* m_itemsToRead;
};

// Create job queue and add fake jobs to it
void testMultithreadJobExecution()
{
	FakeJobQueue queue( N_THREADS );
	hkCriticalSection itemsLock( 0 );
	hkPseudoRandomGenerator rand( 173 );

	hkArray< FakeClass* > objs( N_ENTITIES );
	for ( int i = 0; i < N_ENTITIES; ++i )
	{		
		objs[i] = new FakeClass;
		objs[i]->m_value = ( hkReal )rand.getRand32();
	}

	for ( int s = 0; s < N_STEPS; ++s )
	{
		int nEntitiesPerThread = N_ENTITIES / N_THREADS;

		for ( int i = 0; i < N_THREADS; ++i )
		{
			hkLocalArray< FakeClass* > tempArray( nEntitiesPerThread );

			for ( int j = 0; j < nEntitiesPerThread; ++j )
			{
				tempArray.pushBack( objs[i * nEntitiesPerThread + j] );
			}			
			queue.appendJob( ReadsAndWritesJob( &objs, tempArray, &itemsLock ) );
		}

		queue.waitForCompletion();
	}

	for ( int s = 0; s < N_STEPS; ++s )
	{
		for ( int i = 0; i < N_THREADS; ++i )
		{		
			for ( int nJobs = 0; nJobs < 20; ++nJobs )
			{
				queue.appendJob( LotsOfReadsJob( &objs ) );
			}
		}

		queue.waitForCompletion();
	}

	for ( int i = 0; i < N_ENTITIES; ++i )
	{		
		objs[i]->removeReference();
	}
}

int multithread_tst_main()
{
#if HK_CONFIG_THREAD==HK_CONFIG_MULTI_THREADED
	testMultithreadJobExecution();
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( multithread_tst_main, "Slow", "Common/Test/UnitTest/Base/", __FILE__ );

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
