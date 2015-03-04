/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Base/Thread/Thread/hkThread.h>

hkStackTracer hkMultiThreadCheck::s_stackTracer;
hkStackTracer::CallTree* hkMultiThreadCheck::s_stackTree = HK_NULL;

hkCriticalSection* hkMultiThreadCheck::m_criticalSection = HK_NULL;

void hkMultiThreadCheck::staticInit(hkMemoryAllocator *memoryAllocator)
{
	m_criticalSection = new hkCriticalSection(1000);

	static hkUlong s_stackTreeMemory[sizeof( hkStackTracer::CallTree )];
	s_stackTree = new ( s_stackTreeMemory ) hkStackTracer::CallTree();

	s_stackTree->init( memoryAllocator );
}

void hkMultiThreadCheck::staticQuit()
{
	if ( s_stackTree )
	{
		s_stackTree->quit();
		s_stackTree->~CallTree();
	}

	if ( m_criticalSection )
	{
		delete m_criticalSection;
		m_criticalSection = HK_NULL;
	}
}

void HK_CALL _printStackTrace( const char* text, void* context )
{
	hkStringBuf* buf = (hkStringBuf*)context;
	hkStringBuf tempBuf = text;
	if ( -1 == tempBuf.indexOf( "Cannot find symbol" ) )
	{
		int index = tempBuf.lastIndexOf( "\\" ); 
		if ( -1 != index )
		{
			tempBuf.chompStart( index + 1 );
		}

		*buf += tempBuf;
	}
}

void hkMultiThreadCheck::markForReadImpl(ReadMode mode )
{
	if ( !isCheckingEnabled() )
	{
		return;
	}

	hkCriticalSectionLock lock( m_criticalSection );

	switch ( m_threadId )
	{
		case hkUint32(MARKED_RO_SELF_ONLY):
			{
				HK_ASSERT2( 0xf032df30, mode != RECURSIVE, "You cannot turn a multithread mark from a read only non recursive mark to a recursive mark" );
			}

		case hkUint32(UNMARKED):
		case hkUint32(MARKED_RO):
			{
				m_threadId = (mode == RECURSIVE)? hkUint32(MARKED_RO) : hkUint32( MARKED_RO_SELF_ONLY );
				break;
			}

		default:
			{
#ifdef HK_DEBUG
				hkUint32 threadId = (hkUint32)( hkThread::getMyThreadId() & 0xffffffff );
				if ( threadId != m_threadId )
				{
					// Retrieve the stack trace matching the code location where this object was locked
					hkUlong trace[5];
					int numTrace = s_stackTree->getCallStack( m_stackTraceId, trace, HK_COUNT_OF( trace ) );

					hkStringBuf stackBuf;
					stackBuf = "Your object is already marked for write by another thread, you have to create a critical section around the havok object to avoid conflicts\n";
					stackBuf += "Stack trace for original mark is:\n";
					s_stackTracer.dumpStackTrace( trace, numTrace, _printStackTrace, &stackBuf );

					HK_ASSERT2( 0xf02132d3, ( threadId == m_threadId ), stackBuf );
				}
#endif
			}
	}

	// Push a read bit onto the stack
	{
		m_markBitStack <<= 1;
		m_markCount++;
		HK_ON_DEBUG(const hkInt32 maxDepth = 8 * sizeof(m_markBitStack));
		HK_ASSERT3( 0x63280484, m_markCount <= maxDepth, "Mark nesting too deep. Marks can only be nested up to a maximum depth of " << maxDepth);
	}
}

void hkMultiThreadCheck::markForWriteImpl( )
{
	if ( !isCheckingEnabled() )
	{
		return;
	}

	hkCriticalSectionLock lock( m_criticalSection );

	HK_ASSERT2( 0xf02de43e, m_threadId != (hkUint32)MARKED_RO && m_threadId != (hkUint32)MARKED_RO_SELF_ONLY, "You cannot turn a markForRead into a markForWrite" );

	const hkUint32 threadId = (hkUint32)( hkThread::getMyThreadId() & 0xffffffff );

	// Push a write mark onto the stack
	{
		m_markBitStack = (m_markBitStack<<1) | 1;
		m_markCount++;
		HK_ON_DEBUG(const hkInt32 maxDepth = 8 * sizeof(m_markBitStack));
		HK_ASSERT3( 0x54479d9c, m_markCount  <= maxDepth, "Mark nesting too deep. Marks can only be nested up to a maximum depth of " << maxDepth);
	}

	if ( m_threadId == (hkUint32)UNMARKED )
	{
#ifdef HK_DEBUG
		// Store the call stack trace. In the event of a conflicting lock, this will identify the code location from where this object was locked.
		hkUlong trace[5];
		int ntrace = s_stackTracer.getStackTrace( trace, HK_COUNT_OF( trace ) );
		m_stackTraceId = s_stackTree->insertCallStack( trace, ntrace );	
#endif
		m_threadId = threadId;
	}
	else
	{
#ifdef HK_DEBUG
		if ( threadId != m_threadId )
		{
			// Retrieve the stack trace matching the code location where this object was locked
			hkUlong trace[5];
			int numTrace = s_stackTree->getCallStack( m_stackTraceId, trace, HK_COUNT_OF( trace ) );

			hkStringBuf stackBuf;
			stackBuf = "Your object is already marked for write by another thread, you have to create a critical section around the havok object to avoid conflicts\n";
			stackBuf += "Stack trace for original mark is:\n";
			s_stackTracer.dumpStackTrace( trace, numTrace, _printStackTrace, &stackBuf );

			HK_ASSERT2( 0xf02132d3, ( threadId == m_threadId ), stackBuf );
		}
#endif
	}
}

bool hkMultiThreadCheck::isMarkedForWriteImpl( )
{
	if ( !isCheckingEnabled() )
	{
		return true;
	}

	hkCriticalSectionLock lock( m_criticalSection );

	const hkUint32 threadId = (hkUint32)( hkThread::getMyThreadId() & 0xffffffff );

	if ( m_threadId == threadId )
	{
		return true;
	}
	return false;
}

void hkMultiThreadCheck::unmarkForReadImpl( )
{
	if ( !isCheckingEnabled() )
	{
		return;
	}
	hkCriticalSectionLock lock( m_criticalSection );


	// Pop a lock from the top of the stack and ensure that it is a read lock.
	{
		HK_ASSERT2( 0xf043d534, m_markCount > 0, "Unbalanced mark/unmark: Missing markForRead.  Make sure HK_DEBUG_MULTI_THREADING is defined if mixing Release and Debug libs. See hkMultiThreadCheck.h for more info.");
		HK_ASSERT2( 0x54006467, (m_markBitStack & 1) == 0, "Calling unmark for read on a write mark.");
		m_markCount--;
		m_markBitStack >>= 1;
	}

	if ( m_markCount == 0)
	{
		HK_ASSERT2( 0xf02e32df, hkUint32(MARKED_RO) == m_threadId || hkUint32(MARKED_RO_SELF_ONLY) == m_threadId, "Your object was marked by a different thread");
		m_threadId = (hkUint32)UNMARKED;
	}
}

void hkMultiThreadCheck::unmarkForWriteImpl( )
{
	if ( !isCheckingEnabled() )
	{
		return;
	}
	hkCriticalSectionLock lock( m_criticalSection );

#ifdef HK_DEBUG
	s_stackTree->releaseCallStack( m_stackTraceId );	
#endif

	// Pop a lock from the stack, and ensure that it is a write lock.
	{
		HK_ASSERT2( 0xf043d534, m_markCount > 0, "Unbalanced mark/unmark: Missing markForWrite.  Make sure HK_DEBUG_MULTI_THREADING is defined if mixing Release and Debug libs. See hkMultiThreadCheck.h for more info.");
		HK_ASSERT2( 0x717c3b1d, (m_markBitStack & 1) != 0, "Calling unmark for write on a read mark.");
		m_markCount--;
		m_markBitStack >>= 1;
	}

	HK_ASSERT2( 0xf02e32e0,  ( ((hkUint32)( hkThread::getMyThreadId() & 0xffffffff )) == m_threadId ), "Your object was marked by a different thread");
	if ( m_markCount == 0)
	{
		m_threadId = (hkUint32)UNMARKED;
	}
}

void hkMultiThreadCheck::accessCheck( AccessType type ) const
{
	const hkMultiThreadCheck& lock = *this;
	hkUint32 threadid = lock.m_threadId;
	if ( type == HK_ACCESS_IGNORE || !isCheckingEnabled() )
	{
		return;
	}

	// Note we do not need locking in this case
	HK_ASSERT2( 0xf043d534, threadid != (hkUint32)UNMARKED, "MarkForRead MarkForWrite missing. Make sure HK_DEBUG_MULTI_THREADING is defined if mixing Release and Debug libs. See hkMultiThreadCheck.h for more info."  );

	if ( threadid == (hkUint32)MARKED_RO || threadid == (hkUint32)MARKED_RO_SELF_ONLY)
	{
		HK_ASSERT2( 0xf043d534, type == HK_ACCESS_RO, "Requesting a write access to a read only mark. Make sure HK_DEBUG_MULTI_THREADING is defined if mixing Release and Debug libs. See hkMultiThreadCheck.h for more info."  );
	}
	else
	{
		// now we have a read write lock
		HK_ASSERT2( 0xf02e32e1, ((hkUint32)( hkThread::getMyThreadId() & 0xffffffff )) == threadid, "Your object was write marked by a different thread");
	}
}

void HK_CALL hkMultiThreadCheck::accessCheckWithParent( const hkMultiThreadCheck* parentLock, AccessType parentType, const hkMultiThreadCheck& lock, AccessType type )
{
	//
	//	Check the parent
	//
	if ( parentLock == HK_NULL)
	{
		// if there is no parent the entity is not added to the physics system yet
		return;
	}

	hkUint32 parentId = parentLock->m_threadId;
	if ( !parentLock->isCheckingEnabled() )
	{
		return;
	}

	if ( parentId != (hkUint32)MARKED_RO_SELF_ONLY )
	{
		parentLock->accessCheck( parentType );

		//
		//	Check if the parent already includes the child lock
		//
		if ( parentType >= type )
		{
			return;
		}
	}


		// check for parent
	while(1)
	{
		if ( type == HK_ACCESS_IGNORE )		break;
		if ( !parentLock->isCheckingEnabled() ) break;
		if ( parentId == (hkUint32)UNMARKED) break;
		if ( parentId == (hkUint32)MARKED_RO_SELF_ONLY ) break;
		if ( parentId == (hkUint32)MARKED_RO )
		{
			// now we have a read lock in the parent
			if ( type == HK_ACCESS_RO )
			{
				return;	// have read lock, requesting read lock -> ok
			}
			break;
		}
		// now we have a read write lock in the parent
		HK_ASSERT2( 0xf02e32e2, ((hkUint32)( hkThread::getMyThreadId() & 0xffffffff )) == parentId, "Your object was write marked by a different thread");
		return;
	}

	//
	// Check the child
	//
	lock.accessCheck( type );
}


void hkMultiThreadCheck::disableChecks()
{
	m_markCount = m_markCount | 0x8000;
}

void hkMultiThreadCheck::enableChecks()
{
	if ( isCheckingEnabled() )
	{
		return;
	}
	m_threadId = (hkUint32)UNMARKED;
	m_markCount = 0;
}

void hkMultiThreadCheck::reenableChecks()
{
	m_markCount = ( m_markCount & 0x7fff );
}

void hkMultiThreadCheck::globalCriticalSectionLock()
{
#ifdef HK_DEBUG_MULTI_THREADING
	if ( m_criticalSection )
	{
		m_criticalSection->enter();
	}
#endif
}

void hkMultiThreadCheck::globalCriticalSectionUnlock()
{
#ifdef HK_DEBUG_MULTI_THREADING
	if ( m_criticalSection )
	{
		m_criticalSection->leave();
	}
#endif
}

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
