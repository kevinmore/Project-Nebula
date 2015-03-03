/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/StringMap/hkStringMap.h>
#include <Common/Base/Monitor/MonitorStreamAnalyzer/hkMonitorStreamAnalyzer.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>

static const char* HK_CAPTURE_FRAME_STRING = "Fd";
static const char* HK_CAPTURE_PARAMS_STRING = "Ii";

hkReal hkMonitorStreamAnalyzer::g_lastFrameTime = 16666.6f;

#ifndef HK_PLATFORM_SIM_SPU
hkMonitorStreamStringMap::hkMonitorStreamStringMap()
{
}

hkMonitorStreamStringMap::hkMonitorStreamStringMap(hkFinishLoadedObjectFlag f) : m_map(f)
{
}

hkMonitorStreamStringMap::StringMap::StringMap()
{
}

hkMonitorStreamStringMap::StringMap::StringMap(hkFinishLoadedObjectFlag f) : m_string(f)
{
}

//
// Node constructors and destructor
//
hkMonitorStreamAnalyzer::Node::Node(Node* parent, const char* name, NodeType type )
: m_parent(parent), m_name( name ), m_userFlags(0), m_type( type )
{
	if (parent)
	{
		parent->m_children.pushBack(this);
	}
	for (int i = 0; i < NUM_VALUES; i++)
	{
		m_value[i] = 0; m_count[i] = 0;
	}
	m_absoluteStartTime = 0;
}

hkMonitorStreamAnalyzer::Node::~Node()
{
	for (int i=0; i< m_children.getSize();i++)
	{
		delete m_children[i];
	}
	m_children.clear();
}

char* hkMonitorStreamAnalyzer::getStreamBegin()
{
	// If there have been no calls to HK_TIMER_BEGIN, the size of m_data will
	// be 0. Calling m_data[0] would cause an assert in such a case.
	if (m_data.getSize() == 0)
	{
		return HK_NULL;
	}
	else
	{
		return &m_data[0];
	}
}



//
// Monitor stream analyzer constructor and destructor
//
hkMonitorStreamAnalyzer::hkMonitorStreamAnalyzer( int memorySize, int numThreads, int numSpus )
{
	g_lastFrameTime = 16666.6f;

	m_data.reserve( memorySize );
	m_nodeIdForFrameOverview = "Physics 2012";
	resetNumThreads( numThreads, numSpus );
}

void hkMonitorStreamAnalyzer::resetNumThreads( int numThreads, int numSpus )
{
	m_frameInfos._setSize( hkContainerHeapAllocator().get(&m_frameInfos), numThreads + numSpus );
	m_numSpus = numSpus;
	m_numThreads = numThreads;
	reset();
	HK_ASSERT2(0x9876fa45, numThreads + numSpus > 0 && numThreads + numSpus <= 8, "Num threads out of bounds" );
}

void hkMonitorStreamAnalyzer::reset()
{
	m_data.clear();

	for (int i = 0; i < m_frameInfos.getSize(); ++i)
	{
		m_frameInfos[i].clear();
	}
}


hkMonitorStreamFrameInfo::hkMonitorStreamFrameInfo() 
: 	m_heading( "Unknown Heading" ),
	m_indexOfTimer0(0), 
	m_indexOfTimer1(1), 
	m_absoluteTimeCounter(ABSOLUTE_TIME_TIMER_0), 
	m_timerFactor0(1.0f), 
	m_timerFactor1(1.0f),
	m_threadId(0),
	m_frameStreamStart(0),
	m_frameStreamEnd(0)
{
}

union b4Uint32
{
	hkUint8 b[4];
	hkUint32 d;
};

union b4Uint64
{
	hkUint8 b[8];
	hkUint64 d;
};

union b4Float
{
	hkUint8 b[4];
	hkReal d;
};

inline void _byteSwapUint64( hkUint64& v )
{
	union b4Uint64 dataIn, dataOut;
	dataIn.d = v;
	dataOut.b[0] = dataIn.b[7];
	dataOut.b[1] = dataIn.b[6];
	dataOut.b[2] = dataIn.b[5];
	dataOut.b[3] = dataIn.b[4];
	dataOut.b[4] = dataIn.b[3];
	dataOut.b[5] = dataIn.b[2];
	dataOut.b[6] = dataIn.b[1];
	dataOut.b[7] = dataIn.b[0];
	v = dataOut.d;
}

inline void _byteSwapUint32( hkUint32& v )
{
	union b4Uint32 dataIn, dataOut;
	dataIn.d = v;
	dataOut.b[0] = dataIn.b[3];
	dataOut.b[1] = dataIn.b[2];
	dataOut.b[2] = dataIn.b[1];
	dataOut.b[3] = dataIn.b[0];
	v = dataOut.d;
}

#ifdef HK_ENABLE_64BIT_TIMERS
#	define _byteSwapTimerValue _byteSwapUint64
#else
#	define _byteSwapTimerValue _byteSwapUint32
#endif

inline void _byteSwapFloat( float& v )
{
	union b4Float dataIn, dataOut;
	dataIn.d = v;
	dataOut.b[0] = dataIn.b[3];
	dataOut.b[1] = dataIn.b[2];
	dataOut.b[2] = dataIn.b[1];
	dataOut.b[3] = dataIn.b[0];
	v = float(dataOut.d);
} 

void hkMonitorStreamAnalyzer::applyStringMap( const char* frameStart, const char* frameEnd, const hkPointerMap<const void*, const char*>& map, hkBool endianSwap )
{
//	hkPointerMap<void*, char*> map; 
//	for ( int i = 0; i < m_mappings.getSize(); ++i )
//	{
//		const void* ptr = reinterpret_cast<void*>( m_mappings[i].m_id );
//		map.insert(const_cast<void*>(ptr), const_cast<char*>(m_mappings[i].m_string));
//	}

	// if the data came from a different endian machine, we need to run through it and do a quick swap on
	// data that is > byte sized.

	char* current = const_cast<char*>( frameStart );
	char* end = const_cast<char*>( frameEnd );
	while(current < end) // for all frames
	{
		hkMonitorStream::Command* command = reinterpret_cast<hkMonitorStream::Command*>(current);

		// see if we have hit the spu num HACK
		if (!endianSwap && (hkUlong(command->m_commandAndMonitor) < HK_JOB_TYPE_MAX) )
		{
			int* elfId = (int*)command;
			current = (char*)(elfId + 1);
			continue;
		}
		else if (endianSwap) 
		{
			int* elfId = (int*)command;
			hkUint32 elfIdVal = (hkUint32)*elfId;
			_byteSwapUint32( elfIdVal );
			if (elfIdVal < HK_JOB_TYPE_MAX) 
			{
				*elfId = (int)elfIdVal; // Store the endian swapped version so funcs called after this are ok
				current = (char*)(elfId + 1);
				continue;
			}
		}

		// Replace char* with pointer to loaded string
		hkResult res = map.get( command->m_commandAndMonitor, &command->m_commandAndMonitor ); 
		if ( res == HK_FAILURE )
		{
			HK_WARN(0x032154, "Got an unknown string in the command stream. Abortring.");
			return;
		}
	
		switch(command->m_commandAndMonitor[0])
		{
		case 'T': // timer begin
		case 'E': // timer end
		case 'S': // split list
		case 'l': // list end
			{
				hkMonitorStream::TimerCommand* timerCommand = reinterpret_cast<hkMonitorStream::TimerCommand*>( current );
				current = (char*)(timerCommand + 1);
				if (endianSwap)
				{
					_byteSwapTimerValue( timerCommand->m_time0 );
#if !defined(HK_PLATFORM_HAS_SPU)
					_byteSwapTimerValue( timerCommand->m_time1 );
#endif
				}
				break;
			}

		case 'O': // object name
			{
				hkMonitorStream::TimerBeginObjectNameCommand* com = reinterpret_cast<hkMonitorStream::TimerBeginObjectNameCommand*>( current );
				current = (char*)(com+1);

				com->m_objectName = map.getWithDefault( com->m_objectName, "Unknown object name" );

				if (endianSwap)
				{
					_byteSwapTimerValue( com->m_time0 );
#if !defined(HK_PLATFORM_HAS_SPU)
					_byteSwapTimerValue( com->m_time1 );
#endif
				}
				break;
			}

		case 'L': // timer list begin
			{
				hkMonitorStream::TimerBeginListCommand* timerCommand = reinterpret_cast<hkMonitorStream::TimerBeginListCommand*>( current );
				current = (char*)(timerCommand + 1);
				
				HK_ON_DEBUG( hkResult res2 = ) map.get( timerCommand->m_nameOfFirstSplit, &timerCommand->m_nameOfFirstSplit ); 
				HK_ASSERT(0x4bf11ff3,res2 != HK_FAILURE);
				
				if (endianSwap)
				{
					_byteSwapTimerValue( timerCommand->m_time0 );
#if !defined(HK_PLATFORM_HAS_SPU)
					_byteSwapTimerValue( timerCommand->m_time1 );
#endif
				}
				break;
			}

		case 'M':
			{
				hkMonitorStream::AddValueCommand* serializedCommand = reinterpret_cast<hkMonitorStream::AddValueCommand*>( current );
				current = (char*)(serializedCommand + 1);
				if (endianSwap)
				{
					_byteSwapFloat( serializedCommand->m_value );
				}
				break;
			}
		case 'P':
		case 'p':
			{
				hkMonitorStream::Command* serializedCommand = reinterpret_cast<hkMonitorStream::Command*>( current );
				current = (char*)(serializedCommand + 1);
				break;
			}

		case 'F':	// new frame
		case 'N':	// nop, skip command
			{
				hkMonitorStream::Command* com = reinterpret_cast<hkMonitorStream::Command*>(current);
				current = (char*)(com + 1);
				break;
			}
		
		default:
			HK_ASSERT2(0x3f2fecd9, 0, "Inconsistent Monitor capture data" ); return;
		}
	}
}

#endif // !SIMSPU

void hkMonitorStreamAnalyzer::extractStringMap( const char* frameStart, const char* frameEnd, hkPointerMap<const void*, const char*>& map )
{
	const char* current = frameStart;
	const char* end = frameEnd;

	while ( current < end )
	{
		const hkMonitorStream::Command* command = reinterpret_cast<const hkMonitorStream::Command*>( current );

		if ( hkUlong(command->m_commandAndMonitor) < HK_JOB_TYPE_MAX)
		{
			int* elfId = (int*)command;
			current = (char*)(elfId + 1);
			continue;
		}

		char* str = const_cast<char*>(command->m_commandAndMonitor);
		map.insert(str, str);

		switch( command->m_commandAndMonitor[0] )
		{
		case 'S':		// split list
		case 'E':		// timer end
		case 'l':		// list end
		case 'T':		// timer begin
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( current );
				current = (const char *)(com+1);
				break;
			}
		case 'O': // object name
			{
				const hkMonitorStream::TimerBeginObjectNameCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginObjectNameCommand*>( current );
#if !defined(HK_PLATFORM_SIM_SPU)
				map.insert(const_cast<char*>(com->m_objectName), const_cast<char*>(com->m_objectName) );
#endif
				current = (const char*)(com+1);
				break;
			}
		
		case 'L':		// timer list begin
			{
				const hkMonitorStream::TimerBeginListCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginListCommand*>( current );
				map.insert(const_cast<char*>(com->m_nameOfFirstSplit), const_cast<char*>(com->m_nameOfFirstSplit) );
				current = (const char *)(com+1);
				break;
			}

		case 'M':
			{
				const hkMonitorStream::AddValueCommand* com = reinterpret_cast<const hkMonitorStream::AddValueCommand*>( current );
				current = (const char *)(com+1);
				break;
			}
		case 'P':
		case 'p':
		case 'N':
			{
				const hkMonitorStream::Command* com = reinterpret_cast<const hkMonitorStream::Command*>( current );
				current = (const char *)(com+1);
				break;
			}
		
			default:
			HK_ASSERT2(0x5120d10a, 0, "Inconsistent Monitor capture data" ); 	return;
		}
	}
}

#ifndef HK_PLATFORM_SIM_SPU

hkBool hkMonitorStreamAnalyzer::captureFrameDetails( const char* monitorStreamBegin, const char* monitorStreamEnd, hkMonitorStreamFrameInfo& info )
{
	// make sure there is enough capacity for the actual capture data (+params +end of frame)
	int size = static_cast<int>( monitorStreamEnd - monitorStreamBegin); 
	if( (m_data.getCapacity() - m_data.getSize()) < size  )
	{
		HK_WARN_ONCE(0x98efaed7, "Out of memory in the monitor stream analyzer. Timer data will not be captured");
		return false;
	}

	HK_ASSERT2(0xaceca543, info.m_threadId < m_frameInfos.getSize(), "hkMonitorStreamAnalyzer not initialized to the correct number of threads." );
	// copy the frame info
	hkMonitorStreamFrameInfo& newInfo = m_frameInfos[info.m_threadId]._expandOne( hkContainerHeapAllocator().get(&m_frameInfos) );
	newInfo = info;
	newInfo.m_frameStreamStart = m_data.getSize();
	newInfo.m_frameStreamEnd = 	newInfo.m_frameStreamStart + size;

	// allocate the space
	char* data = m_data.expandByUnchecked( size  );

	// copy the capture data
	hkString::memCpy( data , monitorStreamBegin, size );
	

	return true;
}


//
//  build tree
//

static hkMonitorStreamAnalyzer::Node* createNewNode( hkMonitorStreamAnalyzer::Node* parent, const char* name, hkMonitorStreamAnalyzer::Node::NodeType type, bool wantNodeReuse )
{
	// see if we have added this node already to this parent. If so reuse and augment
	// to prevent needless splitting.
	if(wantNodeReuse)
	{
		for (int c=0; parent && name && (c < parent->m_children.getSize()); ++c)
		{
			if (parent->m_children[c]->m_name && (hkString::strCmp(parent->m_children[c]->m_name, name) == 0) )
			{
				return parent->m_children[c];
			}
		}
	}

	return new hkMonitorStreamAnalyzer::Node( parent, name, type );
}



void hkMonitorStreamAnalyzer::Node::setTimers( const hkMonitorStreamFrameInfo& frameInfo, const hkMonitorStream::TimerCommand& start, const hkMonitorStream::TimerCommand& end)
{
	int id0 = frameInfo.m_indexOfTimer0;
	// note we use += here as we may be sharing a previous node.
	if ( id0 >= 0 && id0 < NUM_VALUES)
	{
		if (end.m_time0 < start.m_time0) // wrap around
		{
			#ifdef HK_ENABLE_64BIT_TIMERS
				this->m_value[id0] += frameInfo.m_timerFactor0 * float( end.m_time0 + ( 0xffffffffffffffffULL - start.m_time0 ) );
			#else
				this->m_value[id0] += frameInfo.m_timerFactor0 * float( end.m_time0 + ( 0xffffffff - start.m_time0 ) );
			#endif
		}
		else
		{
			this->m_value[id0] += frameInfo.m_timerFactor0 * float( end.m_time0 - start.m_time0 );
		}
		this->m_count[id0] += 1;
	}
#if !defined(HK_PLATFORM_HAS_SPU)
	int id1 = frameInfo.m_indexOfTimer1;
	if ( id1 >= 0 && id1 < NUM_VALUES)
	{
		if (end.m_time0 < start.m_time0) // wrap around
		{
			#ifdef HK_ENABLE_64BIT_TIMERS
				this->m_value[id1] += frameInfo.m_timerFactor1 * float( end.m_time1 + ( 0xffffffffffffffff - start.m_time1 ) );
			#else
				this->m_value[id1] += frameInfo.m_timerFactor1 * float( end.m_time1 + ( 0xffffffff - start.m_time1 ) );
			#endif
		}
		else
		{
			this->m_value[id1] += frameInfo.m_timerFactor1 * float( end.m_time1 - start.m_time1 );
		}
		this->m_count[id1] += 1;
	}
#endif

	if (frameInfo.m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0)
	{
		//HK_ASSERT(0x70c5e11d, m_absoluteStartTime == 0); 
		m_absoluteStartTime = double(frameInfo.m_timerFactor0) * double( start.m_time0 );
	}
#if !defined(HK_PLATFORM_HAS_SPU)
	else if (frameInfo.m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_1)
	{
		//HK_ASSERT(0x1a153430, m_absoluteStartTime == 0);
		m_absoluteStartTime = double(frameInfo.m_timerFactor1) * double( start.m_time1 );
	}
#endif
}

static bool findChildWithHint(hkMonitorStreamAnalyzer::Node* mainNode, const char* childName, int& childId )
{
	if ( (mainNode->m_children.getSize() > childId) && (hkString::strCmp(mainNode->m_children[childId]->m_name, childName) == 0) )
	{
		return true;
	}

	for (int i = 0; i < mainNode->m_children.getSize(); ++i)
	{
		if ( hkString::strCmp(mainNode->m_children[i]->m_name, childName) == 0)
		{
			childId = i;
			return true;
		}
	}
	return false;
}

static void HK_CALL reduceMainTree( hkMonitorStreamAnalyzer::Node* node, int destTimerId, hkReal smoothingFactor )
{
	node->m_value[destTimerId] *= smoothingFactor;
	node->m_count[destTimerId] = 0;
	for ( int i = 0; i < node->m_children.getSize(); ++i)
	{
		reduceMainTree( node->m_children[i], destTimerId, smoothingFactor );
	}
}


static void HK_CALL mergeSubTree( hkMonitorStreamAnalyzer::Node* mainTree, hkMonitorStreamAnalyzer::Node* subTree, int destTimerId, int sourceTimerId, hkReal smoothingFactor )
{
	if (destTimerId >= hkMonitorStreamAnalyzer::NUM_VALUES)
	{
		HK_WARN_ONCE(0x6945fade, "Too many threads for monitor stream analyser.");
		destTimerId = hkMonitorStreamAnalyzer::NUM_VALUES - 1;
	}

	int childId = 0; // Also use this as a hint to hasChild - usually the children will be in the same order.

	for ( int i = 0; i < subTree->m_children.getSize(); ++i )
	{
		hkMonitorStreamAnalyzer::Node* matchedNode = HK_NULL;
		hkMonitorStreamAnalyzer::Node* subTreeNode = subTree->m_children[i];

		if ( !findChildWithHint( mainTree, subTree->m_children[i]->m_name, childId ) )
		{
			// No child exists in the main tree - add a new one
			matchedNode = new hkMonitorStreamAnalyzer::Node(mainTree, subTreeNode->m_name, subTreeNode->m_type);
		}
		else
		{
			matchedNode = mainTree->m_children[childId];
		}
		matchedNode->m_value[destTimerId] += (1.0f - smoothingFactor) * subTreeNode->m_value[sourceTimerId];
		matchedNode->m_count[destTimerId] = subTreeNode->m_count[sourceTimerId];
		mergeSubTree(matchedNode, subTreeNode, destTimerId, sourceTimerId, smoothingFactor );

		if (childId < subTree->m_children.getSize() - 1 )
		{
			childId++;
		}
	}
}



void hkMonitorStreamAnalyzer::mergeTreesForCombinedThreadSummary( hkMonitorStreamAnalyzer::Node* mainTree, hkMonitorStreamAnalyzer::Node* subTree, int destTimerId, int sourceTimerId, hkReal smoothingFactor )
{
	reduceMainTree( mainTree, destTimerId, smoothingFactor );
	mergeSubTree( mainTree, subTree, destTimerId, sourceTimerId, smoothingFactor );
}


hkMonitorStreamAnalyzer::Node* hkMonitorStreamAnalyzer::navigateMonitors( const hkMonitorStreamAnalyzer::CursorKeys& keys, hkMonitorStreamAnalyzer::Node* activeNodeIn )
{
	hkMonitorStreamAnalyzer::Node* activeNode = activeNodeIn;

	if ( activeNode == HK_NULL )
	{
		return HK_NULL;
	}

	if ( keys.m_upPressed )
	{
		hkMonitorStreamAnalyzer::Node* f = activeNode->m_parent->m_children[0];
		// If there is a previous child to go to, move to it
		if ( f != activeNode )
		{
			// find the previous child
			for (int i= 0; i < activeNode->m_parent->m_children.getSize(); ++i)
			{
				if ( activeNode->m_parent->m_children[i] == activeNode)
				{
					f = activeNode->m_parent->m_children[i - 1];
					break;
				}
			}

			// if this value is unfolded, go into it
			while (true)
			{
				if ( ( f->m_children.getSize() > 0) && f->m_userFlags & 1 )
				{
					f = f->m_children[f->m_children.getSize() - 1];
					continue;
				}
				break;
			}
			activeNode = f;
		}
		else
		{
			if ( activeNode->m_parent->m_parent )
			{
				activeNode = activeNode->m_parent;
			}	
		}
	}

	// test for down
	if ( keys.m_downPressed )
	{
		hkMonitorStreamAnalyzer::Node* f  = activeNode;
		if ( ( f->m_children.getSize() > 0 ) && f->m_userFlags & 1 )
		{
			activeNode = f->m_children[0];
		}
		else
		{
			bool foundChild = false;
			while (!foundChild)
			{
				for (int i = 0; i < f->m_parent->m_children.getSize(); ++i)
				{
					if ( (f->m_parent->m_children[i] == f) && i < (f->m_parent->m_children.getSize() - 1) )
					{
						activeNode = f->m_parent->m_children[i + 1];
						foundChild = true;
						break;
					}
				}
				if ( !foundChild && f->m_parent->m_parent )
				{
					f = f->m_parent;
					continue;
				}
				break;
			}
		}
	}

	// test for left
	if ( keys.m_leftPressed )
	{
		if ( activeNode->m_userFlags & 1 )
		{
			activeNode->m_userFlags &= ~1;
		}
		else
		{
			if ( activeNode->m_parent->m_parent )
			{
				activeNode = activeNode->m_parent;
				activeNode->m_userFlags &= ~1;
			}
		}
	}

	// test for right
	if ( keys.m_rightPressed )
	{
		if ( 0 == (activeNode->m_userFlags & 1))
		{
			activeNode->m_userFlags |= 1;
		}
		{
			// unfold the whole subtree
			activeNode->m_userFlags |= 1;
		}
	}

	return activeNode;
}

int hkMonitorStreamAnalyzer::findMaxTimerNameIndent( hkMonitorStreamAnalyzer::Node* node, int recursionDepth, int spacing, bool displayPartialTree )
{
	int indent = spacing * recursionDepth + hkString::strLen(node->m_name) + 8; // Add a few characters - we don't know the (count) value which will be displayed after the timer

	if( !displayPartialTree || ( node->m_userFlags & 1 ) )
	{
		for ( int i = 0; i < node->m_children.getSize(); ++i )
		{
			int childIndent = findMaxTimerNameIndent(  node->m_children[i], recursionDepth + 1, spacing, displayPartialTree );
			if ( childIndent > indent )
			{
				indent = childIndent;
			}
		}
	}
	return indent;

}

void hkMonitorStreamAnalyzer::showCombinedThreadSummaryRec( hkOstream& os, hkMonitorStreamAnalyzer::Node* node, int recursionDepth, int numThreads, int numSpus, const CombinedThreadSummaryOptions& options  )
{
	hkArray<char> textLine;
	hkOstream lineOs( textLine );

	int numFields = numThreads + numSpus;

	if (recursionDepth != 0)
	{
		{
			// Show arrow for active node if only displaying part of the tree

			if (options.m_displayPartialTree)
			{
				lineOs << char((options.m_activeNode == node) ? options.m_rightArrowChar  : ' ');
			}

			int extraSpacing = options.m_tabSpacingForTimerNames * (recursionDepth - 1);
			for(int i=0; i < extraSpacing; ++i)		lineOs << ' ';
		}

		// Show expandable tree sub parts if only displaying part of the tree
		if (options.m_displayPartialTree )
		{
			if( node->m_children.getSize() > 0 )
			{
				lineOs << ((node->m_userFlags & 1) ? options.m_downArrowChar : options.m_rightArrowChar );
			}
			else
			{
				lineOs << ' ';
			}
		}

		// Print the timer names, and total count for all threads for each timer
		int count = 0;
		{
			for ( int i = 0; i < numFields; ++i )
			{
				count += node->m_count[i];
			}
		}

		if(options.m_showUnaccountedForTimes && node->m_type == Node::NODE_TYPE_TIMER)
		{
			hkReal	nodeSum = node->valuesSum();
			hkReal	childrenSum = 0;
			for(int i=0; i<node->m_children.getSize(); ++i)
			{
				if(node->m_children[i]->m_type == Node::NODE_TYPE_TIMER)
				{
					childrenSum += node->m_children[i]->valuesSum();
				}
			}
			int		unaccountedFor = 0;
			if(count && childrenSum > 0.0f)
			{
				hkReal	leftOver = nodeSum - childrenSum;
				unaccountedFor = (int) ((leftOver / nodeSum) * 100.0f + 0.5f);
			}

			// You can enable reporting of % time unaccounted for. Disabled for now.
			if(0 && (unaccountedFor >= 1))
			{
				lineOs.printf("%s (%i,~%d%%) ", node->m_name, count, unaccountedFor );
			}
			else
			{
				lineOs.printf("%s (%i) ", node->m_name, count );
			}
		}
		else
		{
			lineOs.printf("%s (%i) ", node->m_name, count );
		}

		// Print timer columns
		for ( int i = 0; i < numFields; ++i )
		{

			if (options.m_useTabsNotSpacesForColumns)
			{
				lineOs << '\t';	
			}
			else
			{	
				int extraSpacing = options.m_indentationToFirstTimerValue + options.m_timerColumnWidth * i  + options.m_tabSpacingForTimerValues * ( recursionDepth - 1) - textLine.getSize();
				for (int j = 0; j < extraSpacing; ++j) lineOs << ' ';
			}

			lineOs.printf("%-6.3f (%i)", node->m_value[i], node->m_count[i] );

		}

		os << textLine.begin();
		os << '\n';
	}

	if( !options.m_displayPartialTree || ( node->m_userFlags & 1 ) )
	{
		for ( int i = 0; i < node->m_children.getSize(); ++i )
		{
			showCombinedThreadSummaryRec( os, node->m_children[i], recursionDepth + 1, numThreads, numSpus, options );
		}
	}
}



hkMonitorStreamAnalyzer::Node* hkMonitorStreamAnalyzer::makeStatisticsTreeForSingleFrame( const char* frameStart, 
																						const char* frameEnd, 
																						const hkMonitorStreamFrameInfo& frameInfo, 
																						const char* rootNodeName,
																						hkBool reuseNodesIfPossible )
{

	const char* currentStreamPtr = frameStart;
	Node* currentNode = new Node(HK_NULL, rootNodeName, hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY );
	Node* rootNode = currentNode;
	Node* lastBeginListNodeCreated = HK_NULL;	// helps debugging
	hkInplaceArray<hkMonitorStream::TimerCommand,16> timerStack;

	// The stack below is just for debugging - it holds all the last timers parsed. If you are missing an 'end' macro
	// then the culprit *is likely* to be one of debugMismatchCheckerTimerStack[0] to debugMismatchCheckerTimerStack[ current size of debugMismatchCheckerTimerStack - 1]
	// and is most likely the most recently altered file. However further monitor parsing may have overwritten the culprit so it may not be in this array.
	// The only way to match things perfectly and always identify the culprit is to use HK_TIMER_NAMED_END
	hkInplaceArray<hkMonitorStream::TimerCommand,16> debugMismatchCheckerTimerStack;

	while( currentStreamPtr < frameEnd ) 
	{

		{
			const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
			if ( hkUlong(com->m_commandAndMonitor) < HK_JOB_TYPE_MAX)
			{
				int* elfId = (int*)com;
				currentStreamPtr = (char*)(elfId + 1);
				continue;
			}
		}

		const char* string = reinterpret_cast<const hkMonitorStream::Command*>(currentStreamPtr)->m_commandAndMonitor;

		//printf("%s\n", string );

		switch(string[0])
		{
		case 'T': // timer begin
			{
				currentNode = createNewNode( currentNode, string + 2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER, reuseNodesIfPossible );
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				timerStack.pushBack( *com );
				debugMismatchCheckerTimerStack.pushBack( *com );
				currentStreamPtr = (const char *)(com+1);
				break;
			}
		case 'E': // timer end
			{
				if ( timerStack.getSize() == 0)
				{
					HK_WARN(0xfafe7975, "Unmatched HK_TIMER_END() macro (with no HK_TIMER_BEGIN()) in timed code");
					return rootNode;
				}
				const hkMonitorStream::TimerCommand& start = timerStack[timerStack.getSize() - 1];
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );

				if ( string[2] && hkString::strCmp( start.m_commandAndMonitor+2, string+2) !=0 )
				{
					HK_WARN( 0xf03edefe, "Unmatched timercommand: '" << start.m_commandAndMonitor+1 << "' =! '" << string+1 );
					return rootNode;
				}

				currentNode->setTimers( frameInfo, start, *com );
				currentNode = currentNode->m_parent;
				timerStack.popBack();
				debugMismatchCheckerTimerStack.setSize( debugMismatchCheckerTimerStack.getSize() - 1 );
				currentStreamPtr = (const char *)(com + 1);
				break;
			}
		case 'O': // object name
			{
				const hkMonitorStream::TimerBeginObjectNameCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginObjectNameCommand*>( currentStreamPtr );
				timerStack.pushBack( *com );
				currentNode = createNewNode( currentNode, com->m_objectName, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER, reuseNodesIfPossible );
				debugMismatchCheckerTimerStack.pushBack( *com );
				currentStreamPtr = (const char *)(com+1);
				break;

			}

		case 'L': // timer list begin
			{
				const hkMonitorStream::TimerBeginListCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginListCommand*>( currentStreamPtr );
				{
					timerStack.pushBack( *com );
					debugMismatchCheckerTimerStack.pushBack( *com );
					currentNode = createNewNode( currentNode, string + 2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER, reuseNodesIfPossible );
					lastBeginListNodeCreated = currentNode;
				}
				{
					hkMonitorStream::TimerCommand com2 = *com;
					com2.m_commandAndMonitor = com->m_nameOfFirstSplit;
					timerStack.pushBack( com2 );
					debugMismatchCheckerTimerStack.pushBack( com2 );
					currentNode = createNewNode( currentNode, com2.m_commandAndMonitor+2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER, reuseNodesIfPossible );
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'S': // split list
			{
				if ( timerStack.getSize() == 0)
				{
					HK_WARN(0xfafe79fa, "Unmatched HK_TIMER_SPLIT_LIST() macro (with no HK_TIMER_BEGIN_LIST()) in timed code");
					return rootNode;
				}
				hkMonitorStream::TimerCommand& start = timerStack[ timerStack.getSize()- 1 ];
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				currentNode->setTimers( frameInfo, start, *com );
				currentNode = createNewNode( currentNode->m_parent, string + 2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER, reuseNodesIfPossible );
				lastBeginListNodeCreated = currentNode;
				start = *com;
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'l': // list end
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				{
					if ( timerStack.getSize() <= 1)
					{
						HK_ON_DEBUG( const char* lastName = (lastBeginListNodeCreated) ? lastBeginListNodeCreated->m_name : "ROOT" );
						HK_WARN(0xfafe79fb, "Unmatched HK_TIMER_END_LIST() macro (with no HK_TIMER_BEGIN_LIST()) in timed code, last beginList created was " << lastName);
						return rootNode;
					}

					const hkMonitorStream::TimerCommand& start = timerStack[ timerStack.getSize()- 1 ];
					if ( start.m_commandAndMonitor[0] != 'S')
					{
						HK_WARN( 0xf0323454, "Mismatched HK_TIMER_END_LIST()/HK_TIMER_BEGIN_LIST() found. Probably missing a HK_TIMER_BEGIN_LIST() *or* timer '" << start.m_commandAndMonitor + 2 << "' has no END macro. Otherwise check debugMismatchCheckerTimerStack in this function for likely culprits.");
						return rootNode;
					}

					currentNode->setTimers( frameInfo, start, *com );
					currentNode = currentNode->m_parent;
					timerStack.popBack();
					debugMismatchCheckerTimerStack.setSize( debugMismatchCheckerTimerStack.getSize() - 1 );
				}
				{
					const hkMonitorStream::TimerCommand& start = timerStack[ timerStack.getSize()- 1 ];
					HK_ASSERT3( 0xf0323454, start.m_commandAndMonitor[0] == 'L', "Mismatched HK_TIMER_END_LIST()/HK_TIMER_BEGIN_LIST() found. Probably missing a HK_TIMER_BEGIN_LIST() *or* timer '" << start.m_commandAndMonitor + 2 << "' has no END macro. Otherwise check debugMismatchCheckerTimerStack in this function for likely culprits.");
					currentNode->setTimers( frameInfo, start, *com );
					currentNode = currentNode->m_parent;
					timerStack.popBack();
					debugMismatchCheckerTimerStack.setSize( debugMismatchCheckerTimerStack.getSize() - 1 );
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'M':
			{
				const hkMonitorStream::AddValueCommand* com = reinterpret_cast<const hkMonitorStream::AddValueCommand*>( currentStreamPtr );
				Node *node = createNewNode( currentNode, string + 2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE, reuseNodesIfPossible);
				node->m_value[0] += com->m_value;
				node->m_count[0] += 1;
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'P':
			{
				const hkMonitorStream::Command* com = reinterpret_cast<const hkMonitorStream::Command*>( currentStreamPtr );
				currentNode = createNewNode( currentNode, string + 2, hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY, reuseNodesIfPossible);
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'p':
			{
				const hkMonitorStream::Command* com = reinterpret_cast<const hkMonitorStream::Command*>( currentStreamPtr );
				currentNode = currentNode->m_parent;
				if ( !currentNode )
				{
					HK_WARN(0xf023df45, "Mismatched HK_TIMER_POP() function");
					return rootNode;
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'F':	// new frame
			{
				HK_ASSERT(0x39129e52,0);
				break;
			}
		case 'N': // nop
			{
				const hkMonitorStream::Command* com = reinterpret_cast<const hkMonitorStream::Command*>( currentStreamPtr );
				currentStreamPtr = (const char*)(com + 1);
				break;
			}
		case 'm': // memory
			{
				const hkMonitorStream::MemoryCommand* com = reinterpret_cast<const hkMonitorStream::MemoryCommand*>( currentStreamPtr );
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		default:
			HK_WARN(0x3d7745e3, "Inconsistent Monitor capture data" );
			return rootNode;
		}
	}

	if ( timerStack.getSize() )
	{
		HK_WARN(0x3d7745e3, "Inconsistent Monitor capture data. Probably missing HK_TIMER_END*s eg. for timer '" << timerStack[timerStack.getSize() - 1].m_commandAndMonitor + 2 << "' Check timerStack in this function for likely culprits." );
	}

	return rootNode;
}


hkMonitorStreamAnalyzer::Node* hkMonitorStreamAnalyzer::makeStatisticsTreeForMultipleFrames( int threadId, hkBool reuseNodesIfPossible )
{
	Node* rootNode = new Node( HK_NULL, "/", hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY );
	rootNode->m_children.setSize(m_frameInfos[threadId].getSize());


	for (int i = 0; i < m_frameInfos[threadId].getSize(); ++i)
	{
		hkMonitorStreamFrameInfo& currentFrameInfo = m_frameInfos[threadId][i];

		const char* start = m_data.begin() + currentFrameInfo.m_frameStreamStart;
		const char* end   = m_data.begin() + currentFrameInfo.m_frameStreamEnd;

		rootNode->m_children[i] = makeStatisticsTreeForSingleFrame(	start, end, currentFrameInfo, currentFrameInfo.m_heading, reuseNodesIfPossible );
	}

	return rootNode;
}



/////////////////////////////////////////////////////////////////////////////////
//
// Text output utilities
//
/////////////////////////////////////////////////////////////////////////////////

	
void hkMonitorStreamAnalyzer::showCombinedThreadSummaryForSingleFrame( Node* node, int numThreads, int numSpus, hkOstream& os, CombinedThreadSummaryOptions& options )
{
	options.m_indentationToFirstTimerValue = findMaxTimerNameIndent( node, 0, options.m_tabSpacingForTimerNames, options.m_displayPartialTree );

	hkArray<char> textLine;
	hkOstream lineOs( textLine );

	// Display table title row
	lineOs.printf("Timer Name");  
	if ( (numThreads > 1) || (numSpus > 0) )
	{
		for ( int i = 0; i < numThreads; ++i )
		{
			if (options.m_useTabsNotSpacesForColumns)
			{
				lineOs << '\t';	
			}
			else
			{	
				int extraSpacing = options.m_indentationToFirstTimerValue + options.m_timerColumnWidth * i - textLine.getSize();
				for (int j = 0; j < extraSpacing; ++j) lineOs << ' ';
			}
			lineOs.printf("Thread %d", i );
		}

		for ( int i = 0; i < numSpus; ++i )
		{
			if (options.m_useTabsNotSpacesForColumns)
			{
				lineOs << '\t';	
			}
			else
			{	
				int extraSpacing = options.m_indentationToFirstTimerValue + options.m_timerColumnWidth * (i + numThreads ) - textLine.getSize();
				for (int j = 0; j < extraSpacing; ++j) lineOs << ' ';
			}
			lineOs.printf("Spu %d", i );
		}
	}
	os << textLine.begin();
	os.printf("\n\n");

	showCombinedThreadSummaryRec(os, node, 0, numThreads, numSpus, options );

}

static HK_FORCE_INLINE void HK_CALL hkMakeSum( hkMonitorStreamAnalyzer::Node* sum, const hkMonitorStreamAnalyzer::Node* node )
{
	for (int j = 0; j < hkMonitorStreamAnalyzer::NUM_VALUES; j++)
	{
		sum->m_value[j]  += node->m_value[j];
		sum->m_count[j]  = sum->m_count[j] + node->m_count[j];
	}
}

// The nodes array is arranged as follows:
// For each thread, there is one element in the list
// The root node has the frame info heading
// T

// This function produces several output lists:
//		- a total hierarchical average
//		- a per frame single node
//		- a summary per frame
//		- the detailed view

// (reordered and with these protos for SN compiler:)
static void HK_CALL hkWriteRec( hkOstream& outstream, hkMonitorStreamAnalyzer::Node* node, int RecDepth, float factor );
static void HK_CALL hkMakeSumRecursive( hkMonitorStreamAnalyzer::Node* sum, hkMonitorStreamAnalyzer::Node* tree );
static hkMonitorStreamAnalyzer::Node* HK_CALL hkFindChildByName( hkMonitorStreamAnalyzer::Node* parent, const char* childName, hkBool searchAnyChild );
static hkMonitorStreamAnalyzer::Node* HK_CALL hkFindNextChildByName( hkMonitorStreamAnalyzer::Node* parent, const char* childName, const hkMonitorStreamAnalyzer::Node* oldChild );



// per name: size
	// disable certain items
// sort by size
// for each elem: find all nodes with a name
	// make child statistics
	// make parent statistics

struct hkMonitorStreamAnalyzerPrintByTypeInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_DEBUG_ALLOCATOR( HK_MEMORY_CLASS_BASE, hkMonitorStreamAnalyzerPrintByTypeInfo );

	hkMonitorStreamAnalyzerPrintByTypeInfo() : m_this( HK_NULL, 0, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER) {}
	bool operator< ( const hkMonitorStreamAnalyzerPrintByTypeInfo& other ) const { return m_this.m_value[1] < other.m_this.m_value[1]; }

	hkMonitorStreamAnalyzer::Node m_this;
	hkArray<hkMonitorStreamAnalyzer::Node>::Debug m_parents;
	hkArray<hkMonitorStreamAnalyzer::Node>::Debug m_children;
};

HK_FORCE_INLINE bool hkCompareInfosBySize( const hkMonitorStreamAnalyzerPrintByTypeInfo* a, const hkMonitorStreamAnalyzerPrintByTypeInfo* b)
{
	return *a < *b;
}

static void HK_CALL hkBuildSizePerName( const hkMonitorStreamAnalyzer::Node* node, hkStringMap<hkMonitorStreamAnalyzerPrintByTypeInfo*>& mapOut )
{
	const char* name = node->m_name;

	hkMonitorStreamAnalyzerPrintByTypeInfo* info = mapOut.getWithDefault( name, HK_NULL );
	if ( !info )
	{
		info = new hkMonitorStreamAnalyzerPrintByTypeInfo;
		info->m_this.m_name = name;
		mapOut.insert( name, info );
	}
	hkMakeSum( &info->m_this, node );

	for (int i = 0; i < node->m_children.getSize(); i++ )
	{
		const hkMonitorStreamAnalyzer::Node* child = node->m_children[i];
		hkBuildSizePerName( child, mapOut );

		// insert child as my child as well
		{
			const char* childName = child->m_name;
			int c;
			hkMonitorStreamAnalyzer::Node* childSum = HK_NULL;
			for (c = info->m_children.getSize()-1; c>=0; c-- )
			{
				childSum = &info->m_children[c];
				if ( hkString::strCmp( childName, childSum->m_name) == 0)
				{
					break;
				}
			}
			if ( c < 0)
			{
				childSum = new (info->m_children.expandBy(1)) hkMonitorStreamAnalyzer::Node( HK_NULL, childName, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER );
			}
			hkMakeSum( childSum, child);
		}
	}

	// insert parent as my child as well
	if ( node->m_parent)
	{
		const char* parentName = node->m_parent->m_name;
		int c;
		hkMonitorStreamAnalyzer::Node* parentSum = HK_NULL;
		for (c = info->m_parents.getSize()-1; c>=0; c-- )
		{
			parentSum = &info->m_parents[c];
			if ( hkString::strCmp( parentName, parentSum->m_name) == 0)
			{
				break;
			}
		}
		if ( c < 0)
		{
			parentSum = new (info->m_parents.expandBy(1)) hkMonitorStreamAnalyzer::Node( HK_NULL, parentName, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER );
		}
		hkMakeSum( parentSum, node->m_parent);
	}
}
static void printSingleNodeValues( hkOstream& outstream, hkReal maxMem, const hkMonitorStreamAnalyzer::Node* node )
{
	hkUint32 maxCount = 0;
	int columns = 0;

	{
		for (int i = 0; i < hkMonitorStreamAnalyzer::NUM_VALUES; i++)
		{
			if ( !node->m_count[i] )
			{
				continue;
			}
			columns ++;
			if ( node->m_count[i] > maxCount){ maxCount = node->m_count[i]; }
		}
	}
	const char* sep = "\t\t\t";

	char buffer[256];

	if ( maxMem > 0 )
	{
		hkString::snprintf( buffer, 200, "%s (%i) %4.1f%%", node->m_name, maxCount, 100.0f * node->m_value[1] / maxMem );
	}
	else
	{
		hkString::snprintf( buffer, 200, "%s (%i)", node->m_name, maxCount);
	}


	outstream.printf("%-34s%s", buffer,sep);

	for (int i = 0; i < hkMonitorStreamAnalyzer::NUM_VALUES; i++)
	{
		if ( !node->m_count[i] )
		{
			continue;
		}
		outstream.printf("% 12.3f: ", node->m_value[i]);
	}
	outstream.printf("\n");
}



static void printStatisticsByType( hkOstream& outstream, hkMonitorStreamAnalyzer::Node* rootNode, hkReal maxImportance = 0.01f )
{
	hkArray<hkMonitorStreamAnalyzerPrintByTypeInfo*>::Temp stats;
	{
		hkStringMap<hkMonitorStreamAnalyzerPrintByTypeInfo*> map;
		hkBuildSizePerName( rootNode, map );
		stats.reserve( map.getSize() );
		for ( hkStringMap<hkMonitorStreamAnalyzerPrintByTypeInfo*>::Iterator i = map.getIterator(); map.isValid(i); i = map.getNext(i))
		{
			stats.pushBackUnchecked( map.getValue(i) );
		}
		hkSort( stats.begin(), stats.getSize(), hkCompareInfosBySize );
	}

	hkReal maxVal = stats.back()->m_this.m_value[1];

	outstream.printf("\n\n");
	outstream.printf("************************************\n" );
	outstream.printf("********** Per Type Times    *******\n" );
	outstream.printf("************************************\n" );
	
	for (int i = stats.getSize()-1; i >=0; i--)
	{
		const hkMonitorStreamAnalyzerPrintByTypeInfo* info = stats[i];
		hkReal val = info->m_this.m_value[1];
		if ( val > maxVal * maxImportance )
		{
			const hkMonitorStreamAnalyzer::Node* node = &info->m_this;
			outstream.printf("\n");
			outstream.printf("************************************\n\n" );
			{
				for (int p = 0; p < info->m_parents.getSize(); p++ )
				{
					outstream.printf("\t\t");		
					printSingleNodeValues( outstream, 0.0f, &info->m_parents[p] );
				}
			}
			printSingleNodeValues( outstream, maxVal, node );
			{
				for (int c = 0; c < info->m_children.getSize(); c++ )
				{
					outstream.printf("\t\t");		
					printSingleNodeValues( outstream, 0.0f, &info->m_children[c] );
				}
			}

		}
		delete info;
	}
}





		// Normally the \a nodes are the timings for each thread, the first level children are the frames
void hkMonitorStreamAnalyzer::writeStatisticsDetails( hkOstream& outstream, hkArrayBase<Node*>& nodes, int numThreads, int numSpus, int reportLevel, const char* nodeIdForFrameOverview, bool showMultithreadedSummary )
{

	// Print the version of Havok that created these statistics.
	outstream.printf("Havok version: %s\n", HAVOK_SDK_VERSION_STRING);

	// The first two analyzes only work for one thread at the moment.
	if (1 || nodes.getSize() == 1 )
	{
		const hkArrayBase<Node*>& childNodes = nodes[0]->m_children;

		// the average value
		hkReal avgValue = 0.0f;
		//
		//	summarize everything
		//

		{
			Node sum( 0, "Sum", hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY ); // sum over all frames
			float f = 1.0f / childNodes.getSize();
			for ( int i = 0; i < childNodes.getSize();i++ )
			{
				hkMakeSumRecursive( &sum, childNodes[i] );
			}
			for ( Node* node = hkFindChildByName( &sum, nodeIdForFrameOverview, true ); node; node = hkFindNextChildByName( &sum, nodeIdForFrameOverview, node ) )
			{
				avgValue += node->m_value[0] * f;
			}
			if ( avgValue <= 0.0f ) { avgValue = 1000.0f; }

			if( reportLevel & REPORT_TOTAL )
			{
				outstream.printf("\n\n");
				outstream.printf("*********************************\n" );
				outstream.printf("********** Total Times    *******\n" );
				outstream.printf("*********************************\n" );
				outstream.printf("Timers are added together\n");
				hkWriteRec( outstream, &sum, 0, f );
			}
		}

		//
		// print each frame in a single line
		//
		if( reportLevel & REPORT_PERFRAME_TIME )
		{
			outstream.printf("\n\n");
			outstream.printf("*********************************\n" );
			outstream.printf("********** Per Frame Time *******\n" );
			outstream.printf("*********************************\n" );
			outstream.printf("Ascii Art all frames overview\n" );

			const int GRAPH_SIZE = 40;
			char buffer[GRAPH_SIZE+10];
			hkString::memSet( buffer, ' ', GRAPH_SIZE );
			buffer[GRAPH_SIZE] = 0;

			for ( int i = 0; i < childNodes.getSize();i++ )
			{
				Node* frameRoot = childNodes[i];
				if ( frameRoot->m_children.getSize() == 0 )
				{
					continue;
				}

				hkReal val = 0.0f;
				const char* nodeName = "Unknown";
				{
					for ( Node* node = hkFindChildByName( frameRoot, nodeIdForFrameOverview, true ); node; node = hkFindNextChildByName( frameRoot, nodeIdForFrameOverview, node ) )
					{
						nodeName = node->m_name;
						val  += node->m_value[0];
					}
				}
				//
				// draw graph
				//
				{
					hkReal relVal = 0.5f * GRAPH_SIZE * val / avgValue;
					int index = int(relVal);
					if ( index < 0) index = 0;
					if (index >=GRAPH_SIZE ) index = GRAPH_SIZE-1;
					char *p = buffer;
					int j = 0;
					for (; j +4 < index; j+=4){ *(p++) = '\t'; }
					int j2 = j;
					for (; j2 < index; j2+=1){ *(p++) = ' '; }
					*(p++) = '#';
					j2++;
					j += 4;
					for (; j2 < j; j2+=1){ *(p++) = ' '; }
					for (; j < GRAPH_SIZE; j+=4){ *(p++) = '\t'; }
					*(p) = 0;
					outstream.printf(buffer);
				}
				outstream.printf("%i %-12s %f\n" , i, nodeName, val );
			}
		}
	}
	if( reportLevel & REPORT_PERFRAME_SUMMARY )
	{
		if (nodes.getSize() == 1 || !showMultithreadedSummary)
		{
			// For each frame
			for ( int i = 0; i < nodes[0]->m_children.getSize(); i++ )
			{
				// For each thread
				for (int j = 0; j < nodes.getSize(); ++j )
				{
					Node* node = nodes[j];
					if ( i < node->m_children.getSize() )
					{
						Node sum(0, "Sum", hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY );
						hkMakeSumRecursive( &sum, node->m_children[i] );
						outstream.printf("\n");
						outstream.printf("****************************************\n" );
						outstream.printf("****** Summary Frame:%i Thread:%i ******\n",i, j );
						outstream.printf("****************************************\n" );
						outstream.printf("%s\n", node->m_children[i]->m_name ); // should be the heading from the frameinfo
						hkWriteRec( outstream, &sum, 0, 1.0f );

						if( reportLevel & REPORT_PERFRAME_PERTYPE )
						{
							printStatisticsByType( outstream, &sum );
						}
					}
				}
			}
		}
		else
		{
			//
			//	Multi threaded summary per frame
			//
			for ( int i = 0; i < nodes[0]->m_children.getSize(); i++ )
			{
				outstream.printf("\n");
				outstream.printf("****************************************\n" );
				outstream.printf("****** Summary Frame:%i ******\n",i);
				outstream.printf("****************************************\n" );
				//outstream.printf("%s\n", node->m_children[i]->m_name ); // should be the heading from the frameinfo

				hkLocalArray<Node*> singleFrameNodes(nodes.getSize());
				singleFrameNodes.setSize(nodes.getSize());

				{
					for (int j = 0; j < nodes.getSize(); ++j )
					{
						singleFrameNodes[j] = new Node( HK_NULL, "", hkMonitorStreamAnalyzer::Node::NODE_TYPE_DIRECTORY );

						// We should really ensure through the interface of adding stream data that this never happens
						// i.e. all streams always submit timer data
						if (nodes[j]->m_children.getSize() > i) 
							hkMakeSumRecursive( singleFrameNodes[j], nodes[j]->m_children[i] );
					}
				}

				CombinedThreadSummaryOptions options;
				options.m_tabSpacingForTimerNames = 4;
				options.m_tabSpacingForTimerValues = 2;
				options.m_timerColumnWidth = 16;
				options.m_displayPartialTree = false;
				options.m_useTabsNotSpacesForColumns = false;
				{
					for (int j = 1; j < singleFrameNodes.getSize(); ++j )
					{
						mergeTreesForCombinedThreadSummary(singleFrameNodes[0], singleFrameNodes[j], j, 0, 0 );
					}
				}

				showCombinedThreadSummaryForSingleFrame( singleFrameNodes[0], numThreads, numSpus, outstream, options );
				if( reportLevel & REPORT_PERFRAME_PERTYPE )
				{
					printStatisticsByType( outstream, singleFrameNodes[0] );
				}
				for (int j = 0; j < nodes.getSize(); ++j )
				{
					delete singleFrameNodes[j];
				}
			}
		}


	}




	//
	//	detailed view
	//
	if( reportLevel & REPORT_PERFRAME_DETAIL )
	{
		// For each frame
		for ( int i = 0; i < nodes[0]->m_children.getSize(); i++ )
		{
			// For each thread
			for (int j = 0; j < nodes.getSize(); ++j )
			{
				Node* node = nodes[j];
				if ( i < node->m_children.getSize() )
				{
					outstream.printf("\n\n");
					outstream.printf("***************************************\n" );
					if (j < numThreads )
					{
						outstream.printf("***** Details Frame-%i Thread:%i ******\n", i, j );
					}
					else
					{
						outstream.printf("***** Details Frame-%i Spu:%i ******\n", i, j - numThreads );
					}

					outstream.printf("***************************************\n" );
					outstream.printf("%s\n", node->m_children[i]->m_name );
					hkWriteRec( outstream, node->m_children[i], 0, 1.0f );
				}
			}
		}
	}
}


static void HK_CALL hkWriteRec( hkOstream& outstream, hkMonitorStreamAnalyzer::Node* node, int RecDepth, float factor )
{
	if( RecDepth )
	{
		for( int j = 1; j < RecDepth; j++ )
		{
			outstream.printf("\t" );
		}

		//
		//	Find maximum count
		//  and the number of columns used
		//
		hkUint32 maxCount = 0;
		int columns = 0;
		{
			for (int i = 0; i < hkMonitorStreamAnalyzer::NUM_VALUES; i++)
			{
				if ( !node->m_count[i] )
				{
					continue;
				}
				columns ++;
				if ( node->m_count[i] > maxCount){ maxCount = node->m_count[i]; }
			}
		}

		
		//
		//	print name
		//
		char buffer[256];
		hkString::snprintf( buffer, 200, "%s(%i)", node->m_name, maxCount);

		if ( maxCount <= 1 )
		{
			hkString::snprintf( buffer, 200, "%s", node->m_name);
		}
		else if ( factor == 1.0f)
		{
			hkString::snprintf( buffer, 200, "%s (%i)", node->m_name, maxCount);
		}
		else
		{
			hkString::snprintf( buffer, 200, "%s (%4.1f)", node->m_name, maxCount * factor);
		}
		outstream.printf("%-32s", buffer);

		for( int j = 5 /*RecDepth*/; j < 8; j++ )
		{
			outstream.printf("\t" );
		}


		//
		//	rescale and print values 
		//
		if ( columns >0 )
		{
			int columnsToPrint = columns;
			if (1)
			{
				for ( int i = 0; i < hkMonitorStreamAnalyzer::NUM_VALUES; i++)
				{
					hkUint32 c = node->m_count[i];
					if ( !c )
					{
						continue;
					}
					hkReal val = node->m_value[i] * factor;
					if ( c < maxCount)
					{
						val *= hkReal(maxCount)/hkReal(c);
					}
					columnsToPrint--;
					if (columnsToPrint)
					{
						outstream.printf("% 12.3f: ", val);
					}
					else
					{
						outstream.printf("% 12.3f\n", val);
					}
				}
			}
		}
		else
		{
			outstream.printf("% 12.3f\n", 0.0f);
		}
	}

	for ( int i = 0; i < node->m_children.getSize(); i++ )
	{
		hkWriteRec( outstream, node->m_children[i], RecDepth + 1, factor );
	}
}

static hkMonitorStreamAnalyzer::Node* HK_CALL hkFindChildByName( hkMonitorStreamAnalyzer::Node* node, const char* childName, hkBool searchAnyChild )
{

	// search child
	for( int j = 0; j < node->m_children.getSize(); j++ )
	{
		if( hkString::strCmp( childName, node->m_children[j]->m_name ) == 0 )
		{
			return node->m_children[j];
		}
	}
	if ( searchAnyChild && node->m_children.getSize()>0 )
	{
		return node->m_children[0];
	}
	return HK_NULL;
}

static hkMonitorStreamAnalyzer::Node* HK_CALL hkFindNextChildByName( hkMonitorStreamAnalyzer::Node* node, const char* childName, const hkMonitorStreamAnalyzer::Node* oldChild )
{
	// search child
	int j;
	for( j = 0; j < node->m_children.getSize(); j++ )
	{
		if ( node->m_children[j] == oldChild )
		{
			break;
		}
	}
	j++;
	for( ; j < node->m_children.getSize(); j++ )
	{
		if( hkString::strCmp( childName, node->m_children[j]->m_name ) == 0 )
		{
			return node->m_children[j];
		}
	}
	return HK_NULL;
}




// This duplicates the structure of tree into sum (combining children with the same parent of the same name)
static void HK_CALL hkMakeSumRecursive( hkMonitorStreamAnalyzer::Node* sum, hkMonitorStreamAnalyzer::Node* tree )
{
	for( int i = 0; i < tree->m_children.getSize(); i++ )
	{
		hkMonitorStreamAnalyzer::Node* childIn = tree->m_children[i];
		hkMonitorStreamAnalyzer::Node* childOut = hkFindChildByName( sum, childIn->m_name, false);

		if( !childOut )
		{
			childOut = new hkMonitorStreamAnalyzer::Node( sum, childIn->m_name, childIn->m_type );
		}
		hkMakeSum( childOut, childIn );
		hkMakeSumRecursive( childOut, childIn );
	}
}

void hkMonitorStreamAnalyzer::writeStatistics( hkOstream& outStream, int reportLevel )
{
	checkAllThreadsCapturedSameNumFrames();

	hkArray<Node*> nodes;

	for (int i = 0; i < m_frameInfos.getSize(); ++i )
	{
		Node* node =  hkMonitorStreamAnalyzer::makeStatisticsTreeForMultipleFrames( i, false );
		nodes.pushBack(node);
	}

	hkMonitorStreamAnalyzer::writeStatisticsDetails( outStream, nodes, m_numThreads, m_numSpus, reportLevel, m_nodeIdForFrameOverview, true );

	for ( int i = 0; i < nodes.getSize();i++ )
	{
		delete nodes[i];
	}
}


namespace
{
	struct Indenter
	{
		hkArray<char>::Debug m_array;

		Indenter()
		{
			m_array.reserve(64);
			m_array.pushBack(0);
		}
		const char* get() const
		{
			HK_ASSERT(0x2d7d3f27, m_array.getSize());
			return m_array.begin();
		}
		void left()
		{
			m_array.popBack();
			m_array.popBack();
			m_array.back() = 0;
		}
		void right()
		{
			m_array.back() = ' ';
			m_array.pushBack(' ');
			m_array.pushBack(0);
		}
		void clear()
		{
			m_array.setSize(1);
			m_array[0] = 0;
		}
	};
}

void hkMonitorStreamAnalyzer::writeRawStatistics( hkOstream& os )
{
	checkAllThreadsCapturedSameNumFrames();
	Indenter indenter;
	os.printf("StatisticsDumpInfo(num_threads=%i, num_spus=%i, num_frames=%i)\n", m_numThreads, m_numSpus, m_frameInfos[0].getSize() );

	for( int threadIndex = 0; threadIndex < m_frameInfos.getSize(); ++threadIndex )
	{
		const hkArrayBase< hkMonitorStreamFrameInfo >& frameInfos = m_frameInfos[threadIndex];
		//for( int frameIndex = frameInfos.getSize()-1; frameIndex < frameInfos.getSize(); ++frameIndex )
		for( int frameIndex = 0; frameIndex < frameInfos.getSize(); ++frameIndex )
		{
			const hkMonitorStreamFrameInfo& frame = frameInfos[frameIndex];

			os.printf("FrameInfo(heading='%s', frame=%i, thread_id=%i, time_counter=%i)\n", frame.m_heading.cString(), frameIndex, frame.m_threadId, (int)frame.m_absoluteTimeCounter);

			const void* cur = m_data.begin() + frame.m_frameStreamStart;
			const void* end   = m_data.begin() + frame.m_frameStreamEnd;

			indenter.clear();
			while( cur < end ) 
			{
				switch( static_cast<const hkMonitorStream::Command*>(cur)->m_commandAndMonitor[0] )
				{
					case 'T': // timer begin
					{
						const hkMonitorStream::TimerCommand* com = static_cast<const hkMonitorStream::TimerCommand*>( cur );
						HK_ASSERT(0x70220c6d, com->m_commandAndMonitor[1] == 't');
						os.printf("%sTimerBegin('%s',%u,%u)\n", indenter.get(), com->m_commandAndMonitor+2, com->m_time0, com->m_time1 );
						indenter.right();
						cur = com + 1;
						break;
					}
					case 'E': // timer end
					{
						const hkMonitorStream::TimerCommand* com = static_cast<const hkMonitorStream::TimerCommand*>( cur );
						HK_ASSERT(0x70220c6d, com->m_commandAndMonitor[1] == 't');
						indenter.left();
						os.printf("%sTimerEnd('%s',%u,%u)\n", indenter.get(), com->m_commandAndMonitor+2, com->m_time0, com->m_time1 );
						cur = com + 1;
						break;
					}
					case 'O': // object name
					{
						HK_ASSERT(0x4eca986c, 0);
						break;
					}
					case 'L': // timer list begin
					{
						const hkMonitorStream::TimerBeginListCommand* com = static_cast<const hkMonitorStream::TimerBeginListCommand*>( cur );
						HK_ASSERT(0x70220c6d, com->m_commandAndMonitor[1] == 't');
						os.printf("%sTimerBegin('%s', %u, %u)\n", indenter.get(), com->m_commandAndMonitor+2, com->m_time0, com->m_time1);
						HK_ASSERT(0x70220c6d, com->m_nameOfFirstSplit[1] == 't');
						os.printf("%sTimerSplit('%s', %u, %u)\n", indenter.get(), com->m_nameOfFirstSplit+2, com->m_time0, com->m_time1);
						indenter.right();
						cur = com + 1;
						break;
					}
					case 'S': // split list
					{
						const hkMonitorStream::TimerCommand* com = static_cast<const hkMonitorStream::TimerCommand*>( cur );
						HK_ASSERT(0x70220c6d, com->m_commandAndMonitor[1] == 't');
						indenter.left();
						os.printf("%sTimerSplit('%s', %u, %u)\n", indenter.get(), com->m_commandAndMonitor+2, com->m_time0, com->m_time1);
						indenter.right();
						cur = com + 1;
						break;
					}
					case 'l': // list end
					{
						const hkMonitorStream::TimerCommand* com = static_cast<const hkMonitorStream::TimerCommand*>( cur );
						HK_ASSERT(0x70220c6d, com->m_commandAndMonitor[1] == 't');
						indenter.left();
						os.printf("%sTimerEnd('%s', %u, %u)\n", indenter.get(), com->m_commandAndMonitor+2, com->m_time0, com->m_time1);
						cur = com + 1;
						break;
					}
					case 'M':
					{
						const hkMonitorStream::AddValueCommand* com = static_cast<const hkMonitorStream::AddValueCommand*>( cur );
						os.printf("%sAddValue(%s,%f)\n", indenter.get(), com->m_commandAndMonitor+1, com->m_value);
						cur = com + 1;
						break;
					}
					case 'P':
					{
						HK_ASSERT(0x7a28795d, 0);
						break;
					}
					case 'p':
					{
						HK_ASSERT(0x43e6b574, 0);
						break;
					}
					case 'F':	// new frame
					{
						HK_ASSERT(0x79d692f3,0);
						break;
					}
					case 'N': // nop
					{
						const hkMonitorStream::Command* com = static_cast<const hkMonitorStream::Command*>( cur );
						cur = com + 1;
						break;
					}
					case 'm': // memory
					{
						const hkMonitorStream::MemoryCommand* com = static_cast<const hkMonitorStream::MemoryCommand*>( cur );
						int size = com->m_sizeAndFlags&0x7fffffff;
						hkBool32 isFree = com->m_sizeAndFlags & ~0x7fffffff;
						const char* what = isFree ? "Free" : "Alloc";
						os.printf("%s%s%s(ptr=0x%p, nbytes=%i)\n", indenter.get(), what, com->m_commandAndMonitor+1, com->m_ptr, size);
						cur = com + 1;
						break;
					}
					default:
					{
						HK_WARN(0x3d7745e3, "Inconsistent Monitor capture data" );
						return;
					}
				}
			}
		}
	}
}



//
// Drawing utilities
//

inline hkMonitorStreamAnalyzer::Node* findChildNode( hkMonitorStreamAnalyzer::Node* currentNode, double sampleTime, int absoluteTimeIndex )
{

	// preselect the children
	int start = 0;
	int end = currentNode->m_children.getSize();
	while ( end - start > 2 )
	{
		int mid = (end+start)>>1;
		hkMonitorStreamAnalyzer::Node* child = currentNode->m_children[mid];
		if ( child->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE )
		{
			break;
		}

		if ( child->m_absoluteStartTime > sampleTime )
		{
			end = mid;
		}
		else
		{
			start = mid;
		}
	}

	for (int i = start; i < end; ++i )
	{
		hkMonitorStreamAnalyzer::Node* child = currentNode->m_children[i];
		if ( child->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE )
		{
			continue;
		}

		double endTime = child->m_absoluteStartTime + child->m_value[absoluteTimeIndex];
		if ( sampleTime <= endTime )
		{
			if ( sampleTime >= child->m_absoluteStartTime )
			{
				return child;
			}
			return HK_NULL;
		}
	}
	return HK_NULL;
}



hkMonitorStreamAnalyzer::Node* getNodeAtSample( hkMonitorStreamAnalyzer::Node* currentNode, hkMonitorStreamAnalyzer::Node* lastHit, double sampleTime, int absoluteTimeIndex )
{
	HK_ASSERT(0x43cedd3e, currentNode->m_absoluteStartTime <= sampleTime);
	if ( lastHit  )
	{
		double 	endTime = lastHit->m_absoluteStartTime + lastHit->m_value[absoluteTimeIndex];
		if ( sampleTime <= endTime )
		{
			currentNode = lastHit;
		}
	}

	while ( currentNode )
	{

		hkMonitorStreamAnalyzer::Node* child = findChildNode( currentNode, sampleTime, absoluteTimeIndex );
		if ( child != HK_NULL )
		{
			currentNode = child;
			continue;
		}
		break;
	}
	return currentNode;
}


static void HK_CALL outputStatsForFrame(	hkMonitorStreamAnalyzer::Node* root, 
																	hkReal startTime, 
																	hkReal timeInc, 
																	int maxFrames,
																	hkArray<hkMonitorStreamAnalyzer::Node*>& timerNodesAtTicks, 
																	int absoluteTimeIndex )
{
	hkMonitorStreamAnalyzer::Node* firstNode = root->m_children[0];
	int fni = 0;
	while(firstNode && (firstNode->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE))
	{	
		++fni;
		firstNode = fni < root->m_children.getSize() ? root->m_children[fni] : HK_NULL;
	}

	hkMonitorStreamAnalyzer::Node* lastNode  = root->m_children.back();
	fni = root->m_children.getSize() - 1;
	while(lastNode && (lastNode->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE))
	{	
		--fni;
		lastNode = fni >= 0 ? root->m_children[fni] : HK_NULL;
	}

	if (!lastNode || !firstNode)
		return;

	// Wait until the current time starts
	// adding null names makes the entry in the texture the background texture
	double currentTimeNorm = firstNode->m_absoluteStartTime - startTime;
	double sampleTimeNorm = 0;
	// renormalize times to around 0 (As when rec a demo that has been running for some time
	// the inc may be say 4.0e-5, whereas time may be in the 1000s, thus never incing at all..
	int maxLoops = maxFrames;
	while ( maxLoops && (currentTimeNorm > sampleTimeNorm) )
	{
		sampleTimeNorm += timeInc;
		timerNodesAtTicks.pushBack(HK_NULL);
		--maxLoops;
	}

	double endTimeNorm = lastNode->m_absoluteStartTime + lastNode->m_value[absoluteTimeIndex] - startTime;

	// reset root (frame, as it normally does not have any proper timer info, in case the lookup needs to come back up to parent)
	root->m_absoluteStartTime = firstNode->m_absoluteStartTime;
	root->m_value[absoluteTimeIndex] = float((endTimeNorm + startTime) - root->m_absoluteStartTime + 1.0f);

	hkMonitorStreamAnalyzer::Node* node = HK_NULL;
	while ( maxLoops && (endTimeNorm > sampleTimeNorm) ) // more to sample
	{
		node = getNodeAtSample( root, node, sampleTimeNorm + startTime, absoluteTimeIndex );
		if (node)
		{
			timerNodesAtTicks.pushBack(node);
		}
		
		sampleTimeNorm += timeInc;
		--maxLoops;
	}

}

/// Setting up a color table.
/// WARNING: The demo framework caches this color table
/// in a file in the root demo directory. So if you 
/// modify this table, you must delete this file!!!!!!!!!
void hkMonitorStreamColorTable::setupDefaultColorTable()
{
	//
	// Physics 2012
	//

	addColor( "Physics 2012",		hkColor::GREY25 );

	// Blue for collision detection
	addColor( "NarrowPhase",		hkColor::CORNFLOWERBLUE );
	addColor( "Broadphase",			hkColor::BLUE );
	addColor( "3AxisSweep",			hkColor::MIDNIGHTBLUE );
	addColor( "WaitForExport",		hkColor::DARKSLATEBLUE );

	// Yellow / brown for single threaded work
	addColor( "SplitIsle",			hkColor::BROWN );
	addColor( "PostCollide",		hkColor::BEIGE );
	addColor( "Maintenance",		hkColor::YELLOW );
	addColor( "InterIsland",		hkColor::YELLOW );
	addColor( "PendingOps",			hkColor::YELLOW );
	addColor( "ConstraintCallbacks",hkColor::YELLOW );
	addColor( "SingleObj",			hkColor::TAN );
	addColor( "TOIs",				hkColor::GOLD );
	addColor( "PostSimCB",			hkColor::WHEAT);

	// Green for solve related jobs
	addColor( "Integrate",			hkColor::GREEN );
	addColor( "Actions",			hkColor::SPRINGGREEN );	
	addColor( "SetupJobs",			hkColor::LIMEGREEN );
	addColor( "BuildJacTask",		hkColor::SEAGREEN );
	addColor( "BuildAccumulators",	hkColor::OLIVE );
	addColor( "BuildJacobians",		hkColor::FORESTGREEN );
	addColor( "Solver",				hkColor::PALEGREEN );
	addColor( "Solve",				hkColor::PALEGREEN );
	addColor( "IntegrateMotions",	hkColor::LIGHTGREEN );
	addColor( "SolverExport",		hkColor::DARKSEAGREEN );

	// Red for locking etc
	addColor( "GetNextJob",				hkColor::DARKRED);
	addColor( "GetNextTask",			hkColor::DARKRED);
	addColor( "FinishJob",				hkColor::DARKRED);
	addColor( "FinishTask",				hkColor::DARKRED);
	addColor( "FinishJobAndGetNext",	hkColor::DARKRED);
	addColor( "FinishTaskAndGetNext",	hkColor::DARKRED);
	addColor( "WaitForSignal",			hkColor::MAROON);
	addColor( "WaitForTasks",			hkColor::MAROON);
	addColor( "LockQueue",				hkColor::RED);
	addColor( "CriticalLock",			hkColor::CRIMSON );	

	// Raycasting
	addColor( "hkSpuMoppRaycastQuery", hkColor::MEDIUMPURPLE);

	// Vehicle
	addColor( "VehicleManager", hkColor::PALEGOLDENROD );
	addColor( "Vehicle", hkColor::WHEAT );
	addColor( "VehicleJob", hkColor::DARKKHAKI );
	addColor( "ApplyVehicleForces", hkColor::KHAKI );

	//
	// Physics
	//

	addColor( "Physics",			0xFFFFB300 );

	addColor( "PreCollide",			hkColor::BEIGE );
	addColor( "DispatchCommands",	hkColor::GOLD );
	addColor( "PostCollideEvent",	hkColor::WHEAT );
	addColor( "BroadPhaseTask",		hkColor::BLUE );
	addColor( "NarrowPhaseTask",	hkColor::CORNFLOWERBLUE );
	addColor( "GatherConstraintsTask",	hkColor::LIMEGREEN );
	addColor( "ConstraintSetupTask",	hkColor::LIMEGREEN );

	addColor( "PreSolve",			hkColor::OLIVE );
	addColor( "SolverTask",			hkColor::PALEGREEN );
	addColor( "SolveConstraints",	hkColor::LIGHTGREEN );
	addColor( "SubIntegrate",		hkColor::LIMEGREEN );
	addColor( "UpdateBodies",		hkColor::DARKSEAGREEN );
	addColor( "PostSolve",			hkColor::OLIVE );
	addColor( "PostSimulateEvent",	hkColor::WHEAT );

	// Waits
	addColor( "WaitForMainThread",		hkColor::DARKRED);
	addColor( "WaitForWorkerThreads",	hkColor::DARKRED);
	addColor( "WaitForOtherTasks",		hkColor::DARKRED);

	// Semaphore release timers. It can be very useful for debugging purposes to see the release count in the timers.
	addColor( "ReleaseSemaphore",		hkColor::MEDIUMPURPLE);
	addColor( "ReleaseSemaphore_1",		hkColor::MEDIUMPURPLE);
	addColor( "ReleaseSemaphore_2",		hkColor::MEDIUMPURPLE);
	addColor( "ReleaseSemaphore_3",		hkColor::MEDIUMPURPLE);
	addColor( "ReleaseSemaphore_4",		hkColor::MEDIUMPURPLE);
	addColor( "ReleaseSemaphore_5+",	hkColor::MEDIUMPURPLE);

	//
	// Animation
	//

	addColor( "hkaQuantizedSampleAndCombineJob",	hkColor::ORCHID);
	addColor( "SampleAndBlend",	hkColor::MEDIUMPURPLE);
	addColor( "hkaCpuSampleAnimationJob",	hkColor::YELLOW);
	addColor( "hkaCpuSampleAndCombineJob",	hkColor::PALEGREEN);
	addColor( "SpuAnim",	hkColor::LIGHTBLUE);
	
	//
	// Cloth
	//

	addColor( "Cloth", hkColor::MEDIUMPURPLE);
	addColor( "Simulate", hkColor::LIGHTGREEN);
	addColor( "Accumulate Actions", hkColor::SPRINGGREEN);
	addColor( "Collide", hkColor::DARKGREEN);
	addColor( "Collide And Solve", hkColor::DARKGREEN);
	addColor( "Input Conversion", hkColor::PINK );
	addColor( "Output Conversion", hkColor::PINK );
	addColor( "Mesh Mesh Deform", hkColor::LIGHTBLUE );
	addColor( "Mesh Bone Deform", hkColor::LIGHTBLUE );
	addColor( "Skin", hkColor::BLUE);
	addColor( "Gather All Vertices", hkColor::YELLOW);
	addColor( "Gather Some Vertices", hkColor::YELLOW);
	addColor( "Copy Vertices", hkColor::YELLOW);
	addColor( "Recalculate Some Normals", hkColor::ORANGE);
	addColor( "Recalculate All Normals", hkColor::ORANGE);
	addColor( "Update Some Vertex Frames (N)", hkColor::ORANGE);
	addColor( "Update Some Vertex Frames (T)", hkColor::DARKORANGE);
	addColor( "Update Some Vertex Frames (TB)", hkColor::DARKORANGE);
	addColor( "Update All Vertex Frames (N)", hkColor::ORANGE);
	addColor( "Update All Vertex Frames (T)", hkColor::DARKORANGE);
	addColor( "Update All Vertex Frames (TB)", hkColor::DARKORANGE);
}


hkColor::Argb hkMonitorStreamColorTable::findColor( const char* color )
{
	for (int i = 0; i < m_colorPairs.getSize(); ++i)
	{
		if ( hkString::strCasecmp(m_colorPairs[i].m_colorName, color) == 0)
		{
			return m_colorPairs[i].m_color;
		}
	}
	return m_defaultColor;
}

void hkMonitorStreamColorTable::addColor( const char* name, hkColor::Argb color )
{
	m_colorPairs.pushBack( ColorPair(name, color) );
}


struct TargaHeader2
{
	unsigned char  IDLength;
	unsigned char  ColormapType;
	unsigned char  ImageType;
	unsigned char  ColormapSpecification[5];
	unsigned short XOrigin;
	unsigned short YOrigin;
	unsigned short ImageWidth;
	unsigned short ImageHeight;
	unsigned char  PixelDepth;
	unsigned char  ImageDescriptor;
};

#define GET_ENDIAN_SWAPPED_16(x)  ((((x) & 0xff) << 8) | (( (x) & 0xff00) >> 8))

bool saveToTGA(int* data, hkOstream& s, int width, int height)
{
	// Header
	TargaHeader2 tga;
	hkString::memSet(&tga, 0, sizeof(tga));
	tga.ImageType  = 2; // raw

#if HK_ENDIAN_BIG
	tga.ImageHeight = (unsigned short)GET_ENDIAN_SWAPPED_16(height);
	tga.ImageWidth = (unsigned short)GET_ENDIAN_SWAPPED_16(width);
	for (int h=0; h< height;++h)
	{
		for (int w=0;w<width; ++w)
		{
			char* datac = (char*)( &data[h*width + w] );
			char r = datac[0];
			char g = datac[1];
			char b = datac[2];
			char a = datac[3];
			datac[0] = a;
			datac[1] = b;
			datac[2] = g;
			datac[3] = r;
		}
	}
#else
	tga.ImageHeight = (unsigned short)height;
	tga.ImageWidth = (unsigned short)width;
#endif

	tga.PixelDepth = (unsigned char)32;
	s.write((char*)&tga, sizeof(tga));

	s.write((char*)data, height * width * 4);
	return true;
}


static hkUint32 number_0[7] = 
{
	0x0000800,
	0x0008080,
	0x0080008,
	0x0080008,
	0x0080008,
	0x0008080,
	0x0000800,
};
static hkUint32 number_1[7] = 
{
	0x0008800,
	0x0080800,
	0x0000800,
	0x0000800,
	0x0000800,
	0x0000800,
	0x0088888,
};
static hkUint32 number_2[7] = 
{
	0x0008880,
	0x0080008,
	0x0000008,
	0x0000080,
	0x0000800,
	0x0008000,
	0x0088888,
};
static hkUint32 number_3[7] = 
{
	0x0088880,
	0x0000008,
	0x0000008,
	0x0000888,
	0x0000008,
	0x0000008,
	0x0088880,
};
static hkUint32 number_4[7] = 
{
	0x0000080,
	0x0000880,
	0x0008080,
	0x0008080,
	0x0088888,
	0x0000080,
	0x0000080,
};
static hkUint32 number_5[7] = 
{
	0x0008888,
	0x0008000,
	0x0008000,
	0x0008880,
	0x0000008,
	0x0000008,
	0x0008880,
};
static hkUint32 number_6[7] = 
{
	0x0000880,
	0x0008000,
	0x0080000,
	0x0080880,
	0x0088008,
	0x0080008,
	0x0008880,
};
static hkUint32 number_7[7] = 
{
	0x0088888,
	0x0000008,
	0x0000080,
	0x0000800,
	0x0008000,
	0x0008000,
	0x0008000,
};

static hkUint32 number_8[7] = 
{
	0x0008880,
	0x0080008,
	0x0080008,
	0x0008880,
	0x0080008,
	0x0080008,
	0x0008880,
};

static hkUint32 number_9[7] = 
{
	0x0008880,
	0x0080008,
	0x0080008,
	0x0008888,
	0x0000008,
	0x0000080,
	0x0088800,
};

static hkUint32* numbers[10] = { &number_0[0],&number_1[0],&number_2[0],&number_3[0],&number_4[0],&number_5[0],&number_6[0],&number_7[0],&number_8[0],&number_9[0]};


static void HK_CALL drawDigit( int nr, int currentY, int outputPixelWidth, int* texture )
{
	hkUint32* pattern = numbers[nr];
	for ( int x =0; x < 8; x++)
	{
		for (int y=0; y < 7; y++)
		{
			if ( (pattern[6-y]<<(4*x))&0xf0000000 )
			{
				texture[ x + y * outputPixelWidth] = 0xFF000000;
			}
		}
	}
}

static void HK_CALL drawNumber( int nr, int currentY, int outputPixelWidth, int* texture )
{
	int x = 0;
	for (int i=1000; i >= 1; i=i/10)
	{
		int digit = (nr/i)%10;
		drawDigit( digit, currentY, outputPixelWidth, texture + x);
		x += 7;
	}
}

static hkMonitorStreamAnalyzer::Node* HK_CALL getNodeSampledAtTick(	hkMonitorStreamFrameInfo& info, 
											int frameIndex, int tick,
											const hkArrayBase<hkMonitorStreamAnalyzer::Node*>& nodes, 
											int maxX,
											hkReal frameTime, 
											hkReal absoluteFrameStartTimes )
{
	hkReal timeIncrement = frameTime / (hkReal)maxX;
	hkArray<hkMonitorStreamAnalyzer::Node*> timerNodesAtTicks;
	timerNodesAtTicks.reserveExactly( maxX );

	hkMonitorStreamAnalyzer::Node* sampledNode  = HK_NULL;

	if (nodes[frameIndex]->m_children.getSize() > 0)
	{
		timerNodesAtTicks.clear();

		HK_ASSERT2(0x8f258165, info.m_absoluteTimeCounter != hkMonitorStreamFrameInfo::ABSOLUTE_TIME_NOT_TIMED, \
				"You cannot draw statistics unless one of your timers is absolute time");

		int absoluteTimeIndex = (info.m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0) ? info.m_indexOfTimer0 : info.m_indexOfTimer1;
		
		outputStatsForFrame(nodes[frameIndex], absoluteFrameStartTimes, timeIncrement, maxX, timerNodesAtTicks, absoluteTimeIndex );
		
		int numSamplesThisFrame = timerNodesAtTicks.getSize();

		if ( tick < numSamplesThisFrame )
		{
			sampledNode = timerNodesAtTicks[tick];
		}
	}

	return sampledNode;
}

static void HK_CALL drawStatistics(	const hkMonitorStreamFrameInfo& info, 
									int frameIndex,
									const hkArrayBase<hkMonitorStreamAnalyzer::Node*>& nodes, 
									int* texture, 
									int height, 
									hkMonitorStreamColorTable& colorTable, 
									int pixelWidth, 
									int maxX, 
									hkReal frameTime, 
									hkReal absoluteFrameStartTimes,
									hkPointerMap<const char*, hkColor::Argb>& unknownColorMap)
{
	hkMonitorStreamAnalyzer::Node* rootNode = nodes[frameIndex];
	if (rootNode->m_children.getSize() == 0)
	{
		return;
	}

	HK_ASSERT2(0x8f258165, info.m_absoluteTimeCounter != hkMonitorStreamFrameInfo::ABSOLUTE_TIME_NOT_TIMED, \
			   "You cannot draw statistics unless one of your timers is absolute time");

	hkPointerMap<const char*, hkColor::Argb> colorMap;
	hkReal timeIncrement = frameTime / (hkReal)maxX;
	hkArray<hkMonitorStreamAnalyzer::Node*> timerNodesAtTicks;
	timerNodesAtTicks.reserveExactly( maxX );			

	{
		int absoluteTimeIndex = (info.m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0) ? 
								info.m_indexOfTimer0 : info.m_indexOfTimer1;
		outputStatsForFrame(rootNode, absoluteFrameStartTimes, timeIncrement, maxX, timerNodesAtTicks, absoluteTimeIndex);
	}
		
	int numSamplesThisFrame = timerNodesAtTicks.getSize();
	
	hkMonitorStreamAnalyzer::Node* prevNode = HK_NULL;
	hkColor::Argb prevColor = hkColor::WHITE;
	for (int j = 0; (j < numSamplesThisFrame) && (j < maxX); j++ )
	{	
		hkMonitorStreamAnalyzer::Node* node = timerNodesAtTicks[j];

		// Leave the background texture when there is no timer info
		if (node == HK_NULL || node->m_type != hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER)
		{
			continue;
		}
		
		hkColor::Argb color;
		if (node == prevNode)
		{
			color = prevColor;
		}
		else
		{
			prevNode = node;

			// Use white for timers not registered in the color map
			color = hkColor::WHITE;

			// For this node - try to match the name of the deepest known timer against
			// the color table.  Use 2 maps to speed the process.
			while (node != HK_NULL)
			{
				const char* name = node->m_name;
				if ( colorMap.get(name, &color) == HK_FAILURE )
				{
					if (unknownColorMap.get(name, &color) != HK_FAILURE )
					{
						// If the color is unknown try the parent
						node = node->m_parent;
						continue;
					}

					// Color not in cached map yet - look it up in the table
					bool colorFound = false;
					for (int i = 0; i < colorTable.m_colorPairs.getSize(); ++i)
					{
						if ( hkString::strCasecmp(colorTable.m_colorPairs[i].m_colorName, name) == 0)
						{
							color = colorTable.m_colorPairs[i].m_color;
							colorFound = true;
							break;
						}
					}
					if ( colorFound )
					{
						colorMap.insert(name, color );
						break; // Found color
					}
					else
					{
						unknownColorMap.insert(name, color);
						node = node->m_parent;
					}
				}
				else
				{
					break; // Found color
				}
			}
			
			// Draw first sample of a node with a darkened version of the color to be able to see the boundaries 
			// between nodes with the same color.
			prevColor = color;
			color = hkColor::darken(color, 1);			
		}
		
		for (int k = 0; k < height; k++)
		{
			texture[ k * pixelWidth + j ] = color;
		}
	}			
}

static inline hkReal __hkMin( hkReal a, hkReal b)
{
	return (a<b)?a:b;
}

static inline hkReal __hkMax( hkReal a, hkReal b)
{
	return (a>b)?a:b;
}

static inline double __hkMinD( double a, double  b)
{
	return (a<b)?a:b;
}

static inline double __hkMaxD( double  a, double  b)
{
	return (a>b)?a:b;
}

void hkMonitorStreamAnalyzer::checkAllThreadsCapturedSameNumFrames() const
{
	int numFrames = m_frameInfos[0].getSize();
	for (int i = 1; i < m_frameInfos.getSize(); ++i)
	{
		if ( m_frameInfos[i].getSize() != numFrames )
		{
			HK_WARN_ONCE(0xbebf8746, "Inconsistent number of captured frames between threads, timer analysis data may be corrupt" );
		}
	}
}

int hkMonitorStreamAnalyzer::ThreadDrawInput::computePerThreadHeightToFit( int textureHeight, int numFrames, int numThreads, int frameGap, int threadGap )
{
	int perFrameHeight = ( textureHeight / numFrames ) - frameGap;
	int perThreadHeight = ( perFrameHeight / numThreads ) - threadGap;
	return perThreadHeight;
}

static inline hkUint32 hkRoundUpPow2(hkUint32 n)
{
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;
}

static void _rebase( hkArray<hkMonitorStreamAnalyzer::Node*>& rootNodes, double hdt, double dt)
{
	for (int n=0; n < rootNodes.getSize(); ++n)
	{
		if (rootNodes[n]->m_type != hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE)
		{
			if (rootNodes[n]->m_absoluteStartTime < hdt)
				rootNodes[n]->m_absoluteStartTime += dt;
		}
		_rebase(rootNodes[n]->m_children, hdt, dt);
	}
}

static bool _needsRebase(hkArray<hkMonitorStreamAnalyzer::Node*>& rootNodes, double halfRange, double& maxSoFar )
{
	for (int n=0; n < rootNodes.getSize(); ++n)
	{
		if (rootNodes[n]->m_type != hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE)
		{
			if (rootNodes[n]->m_absoluteStartTime < halfRange)
			{	
				if ((maxSoFar - rootNodes[n]->m_absoluteStartTime) > halfRange)
				{
					// low abs value, with over half the range to small value, small values are in fact overflows
					return true;
				}
			}

			maxSoFar = __hkMaxD(maxSoFar, rootNodes[n]->m_absoluteStartTime);

		}
		if (_needsRebase(rootNodes[n]->m_children, halfRange, maxSoFar))
			return true;
	}
	return false;
}

void hkMonitorStreamAnalyzer::getTimerLimits(const hkArrayBase<Node*>& nodeList, const ThreadDrawInput& input, hkArray< hkArray< hkMonitorStreamFrameInfo > >& frameInfos,
											 hkReal& maxFrameTime,
											 hkArray<hkReal>& startTimes)
{
	int numThreads = nodeList.getSize();
	int frameEnd = input.m_frameStart + input.m_numFrames;
	int numFramesInStream = nodeList[0]->m_children.getSize();
	int numFrames = input.m_numFrames;

	if (frameEnd > numFramesInStream )
	{
		numFrames = numFramesInStream - input.m_frameStart;
		frameEnd  = numFramesInStream;
	}
	if ( numFrames <= 0)
	{
		return;
	}
	
	// Get first start time for the thread so that they are calibrated to 
	// each other. Can be very noticeable on Xbox360 for instance.
	startTimes.setSize(input.m_numFrames, 0.0f);
	maxFrameTime = 0;

	{
		for (int j = input.m_frameStart; j < frameEnd; ++j)
		{
			int fzero = j -input.m_frameStart;
			int absoluteTimeIndex = (frameInfos[0][j].m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0) ? frameInfos[0][j].m_indexOfTimer0 : frameInfos[0][j].m_indexOfTimer1;
			
			// This for loop accounts for wrap around in the unit32 timers, specifically for the timerbase (the timer value will be ok)
			// First we detect if the range is too much and we have some small timers, if so assume we need rebase
			// Note this limits the amount of time we can time for, but it is long enough for realtime
			{
				bool needRebase = false;
			#ifdef HK_ENABLE_64BIT_TIMERS
				double fullRange = double((hkUint64)0xffffffffffffffffULL) * ((frameInfos[0][j].m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0) ? frameInfos[0][j].m_timerFactor0 : frameInfos[0][j].m_timerFactor1);
			#else
				double fullRange = double((hkUint32)0xffffffff) * ((frameInfos[0][j].m_absoluteTimeCounter == hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0) ? frameInfos[0][j].m_timerFactor0 : frameInfos[0][j].m_timerFactor1);
			#endif
				double halfRange = fullRange * 0.5;
				double maxSoFar = 0.0;
				for (int i = 0; i < numThreads; ++i )
				{
					hkArray<Node*>& threadIframeJ = nodeList[i]->m_children[j]->m_children;
					if (_needsRebase(threadIframeJ,halfRange,maxSoFar))
					{
						needRebase = true;
						break;
					}
				}
				if (needRebase)
				{
					for (int ii = 0; ii < numThreads; ++ii )
					{
						hkArray<Node*>& threadIframeJ = nodeList[ii]->m_children[j]->m_children;
						_rebase(threadIframeJ,halfRange,fullRange);
					}
				}
			}

			hkArray<double> threadStartTimes(numThreads,HK_REAL_MAX);
			for (int i = 0; i < numThreads; ++i )
			{
				hkArray<Node*>& threadIframeJ = nodeList[i]->m_children[j]->m_children;
				// take first node's start time (frame node not normally a proper timed node)
				for (int iv=0; iv < threadIframeJ.getSize(); ++iv)
				{
					if (threadIframeJ[iv]->m_type != Node::NODE_TYPE_SINGLE)
					{
						double st = threadIframeJ[iv]->m_absoluteStartTime;
						threadStartTimes[i] = __hkMinD( st, threadStartTimes[i]);
					}
				}
			}

			startTimes[fzero] = HK_REAL_MAX; 
			for (int si = 0; si < numThreads; ++si )
			{
				startTimes[fzero] = __hkMin( (float)threadStartTimes[si], startTimes[fzero] );
			}

			for (int iii = 0; iii < numThreads; ++iii )
			{
				hkArray<Node*>& threadIframeJ = nodeList[iii]->m_children[j]->m_children;
				for (int iv=0; iv < threadIframeJ.getSize(); ++iv)
				{
					if (threadIframeJ[iv]->m_type != Node::NODE_TYPE_SINGLE)
					{
						double endTime = threadIframeJ[iv]->m_absoluteStartTime + threadIframeJ[iv]->m_value[absoluteTimeIndex] - startTimes[fzero];
						maxFrameTime = __hkMax( float(endTime), maxFrameTime);
					}
				}
			}

		}
	}

	if (input.m_limitStartTime > 0.0f)
	{
		for (int j = input.m_frameStart; j < frameEnd; ++j)
		{
			int fzero = j -input.m_frameStart;
			startTimes[fzero] += input.m_limitStartTime;
		}
	}

	if (input.m_limitFrameTime > 0.0f) 
	{
		maxFrameTime = input.m_limitFrameTime;
	}
	else if (numFrames == 1)
	{
		// Timescale is assumed to be in microseconds i.e. timerFactor = 1e6 / ticksPerSec
		// Adjust frame time slowly if in visual mode
	
		hkReal maxFrameTimeRounded = hkRoundUpPow2( (int)(maxFrameTime / 1000.0f) + 1) * 1000.0f;

		// Never allow more than 25% spike
		maxFrameTime = g_lastFrameTime + __hkMin( g_lastFrameTime / 4,  (maxFrameTimeRounded - g_lastFrameTime) ) * 0.05f;

		//Clamp min to 60Hz frame
		maxFrameTime = __hkMax(16666, maxFrameTime);

		// Only set last when not doing zoom 
		g_lastFrameTime  = maxFrameTime;
	}
}

void hkMonitorStreamAnalyzer::writeStatisticsDetailsToTexture( const hkArrayBase<Node*>& nodeList, const ThreadDrawInput& input, hkArray< hkArray< hkMonitorStreamFrameInfo > >& frameInfos, int*& texture, int& height, SampleInfo* sampleInfo )
{
	int numThreads = nodeList.getSize();
	if (numThreads < 1)
		return; // nothing to do..

	hkReal maxFrameTime = -1.0f;
	hkArray<hkReal> startTimes;
	getTimerLimits( nodeList, input, frameInfos, maxFrameTime, startTimes );



	int numCaptured = frameInfos[0].getSize() - input.m_frameStart;
	int numFrames = input.m_numFrames > numCaptured ? numCaptured : input.m_numFrames;
	int frameEnd = input.m_frameStart + numFrames;

	int pixelHeightPerThread = input.m_heightPerThread + input.m_gapBetweenThreads;
	int pixelHeightPerFrame  = pixelHeightPerThread * numThreads + input.m_gapBetweenFrames;
	int pixelHeight          = pixelHeightPerFrame * numFrames;

	int currentY = pixelHeight;
	int startX   = numFrames > 1 ? 32 : 0 ;
	int maxX = input.m_outputPixelWidth-startX;

	int numTotalPixels = pixelHeight * (input.m_outputPixelWidth + 1);
	texture = hkAllocate<int>(numTotalPixels, HK_MEMORY_CLASS_DEMO);
	height = pixelHeight;
	hkString::memSet(texture, 0x00, numTotalPixels * 4);

	if (sampleInfo)
	{
		sampleInfo->m_maxSampleTime = maxFrameTime;
	}

	hkPointerMap<const char*, hkColor::Argb> unknownColorMap;
	{
		for (int f = input.m_frameStart; f < frameEnd; f++ )
		{
			currentY -= input.m_gapBetweenFrames;

			if (numFrames > 1)
			{
				drawNumber( f, currentY, input.m_outputPixelWidth, &texture[ (currentY-pixelHeightPerThread) * input.m_outputPixelWidth] );
			}

			for (int i = 0; i < numThreads; ++i )
			{
				currentY -= pixelHeightPerThread;

				int* output = &texture[currentY * input.m_outputPixelWidth+startX];
				drawStatistics( frameInfos[i][0], f, nodeList[i]->m_children, output, 
					input.m_heightPerThread, *input.m_colorTable, input.m_outputPixelWidth, maxX, maxFrameTime,
					startTimes[f-input.m_frameStart], unknownColorMap );
			}
		}
	}
	HK_ASSERT( 0xf0212343, currentY == 0);

	//
	//	Draw 1 msec lines
	//
	hkReal pixelsPerMs = (1000 * maxX) / maxFrameTime;
	int pixelOffset = int( hkReal( int(input.m_limitStartTime) % 1000) * 0.001f * pixelsPerMs ); 
	{
		if (pixelsPerMs > 5)
		{
			int x = startX - pixelOffset;
			int numMsBars = 0;
			while ( x < input.m_outputPixelWidth )
			{
				if (x >= 0)
				{
					for (int y=0; y < pixelHeight;y++)
					{
						texture[y * input.m_outputPixelWidth+x] = 0xff0000ff;
					}
				}
				numMsBars++;
				x = startX - pixelOffset + int( numMsBars * pixelsPerMs );
			}
		}
	}

	//
	//	Draw 60Hz frame markers
	//
	{
		int pixelsPerFrame = int(16666 * maxX / maxFrameTime);
		if (pixelsPerFrame > 0)
		{
			for (int x = startX - pixelOffset; x < input.m_outputPixelWidth; x+= pixelsPerFrame)
			{
				if (x < 0) 
					continue;

				for (int y=0; y < pixelHeight;y++)
				{
					texture[y * input.m_outputPixelWidth+x] = 0xff00ff00;
					texture[y * input.m_outputPixelWidth+x+1] = 0xff00ff00;
				}
			}
		}
	}

	if (input.m_warnAboutMissingTimers)
	{
		// Warn about unknown timers
		for (hkPointerMap<const char*, hkColor::Argb>::Iterator itr = unknownColorMap.getIterator(); unknownColorMap.isValid(itr); itr = unknownColorMap.getNext( itr ) )
		{
			HK_WARN(0x94696eee, "Unknown timer when drawing monitor output: " << unknownColorMap.getKey(itr));
		}
	}


}
void hkMonitorStreamAnalyzer::writeStatisticsDetailsToTga( const hkArrayBase<Node*>& nodeList, const hkMonitorStreamAnalyzer::ThreadDrawInput& input, hkArray< hkArray< hkMonitorStreamFrameInfo > >& frameInfos, hkOstream& outStream, SampleInfo* sampleInfo)
{
	int* texture = HK_NULL; 
	int height = 0;

	writeStatisticsDetailsToTexture( nodeList, input, frameInfos, texture, height, sampleInfo  );
	
	if (texture)
	{
		saveToTGA( texture, outStream, input.m_outputPixelWidth, height );
		hkDeallocate(texture);
	}
}

void hkMonitorStreamAnalyzer::drawThreadsToTga( const hkMonitorStreamAnalyzer::ThreadDrawInput& input, hkOstream& outStream )
{
	checkAllThreadsCapturedSameNumFrames();

	hkInplaceArray<Node*, 6> nodeList;
	const int numThreads = m_frameInfos.getSize();

	nodeList.setSize(numThreads);
	for (int i = 0; i < numThreads; ++i)
	{
		nodeList[i] = makeStatisticsTreeForMultipleFrames( i, false );
	}

	writeStatisticsDetailsToTga( nodeList, input, m_frameInfos, outStream );

	for (int ti = 0; ti < numThreads; ++ti)
	{
		delete nodeList[ti];
	}
}


hkMonitorStreamAnalyzer::Node* HK_CALL hkMonitorStreamAnalyzer::reverseLookupNodeAtTgaSample( int x, int y, const hkArrayBase<Node*>& nodeList, const ThreadDrawInput& input, hkArray< hkArray< hkMonitorStreamFrameInfo > >& frameInfos )
{
	int numFrames = input.m_numFrames;
	int numThreads = nodeList.getSize();
	if (numThreads < 1)
		return HK_NULL; // nothing to do..

	int startX   = numFrames > 1 ? 32 : 0 ;
	
	if ( (x < startX) || (x >= input.m_outputPixelWidth)) 
		return HK_NULL; // out of range

	int maxX = input.m_outputPixelWidth-startX;
	x -= startX;

	int pixelHeightPerThread = input.m_heightPerThread + input.m_gapBetweenThreads;
	int pixelHeightPerFrame  = pixelHeightPerThread * numThreads + input.m_gapBetweenFrames;
	int pixelHeight          = pixelHeightPerFrame * numFrames;

	if ( (y < 0) || (y >= pixelHeight))
		return HK_NULL; // out of range

	// frames and threads start at top, so y inverted
	y = (pixelHeight - 1) - y;
	int yFrame = y / pixelHeightPerFrame;
	int yThread = (y % pixelHeightPerFrame) / pixelHeightPerThread;

	hkReal maxFrameTime = -1.0f;
	hkArray<hkReal> startTimes;
	getTimerLimits( nodeList, input, frameInfos, maxFrameTime, startTimes );

	hkMonitorStreamAnalyzer::Node* n = getNodeSampledAtTick( frameInfos[yThread][0], yFrame, x, nodeList[yThread]->m_children, maxX, maxFrameTime, startTimes[yFrame] );

	return n;
}


namespace
{
	/// A structure used in hkMonitorStreamAnalyzer::gatherRawStreamTimerValues() to collect timer values
	struct hkMonitorStreamAnalyzer_TimerCollector
	{
		void HK_FORCE_INLINE init( const char* timerName )
		{
			m_node = hkMonitorStreamAnalyzer::Node( HK_NULL, timerName, hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER );
			m_stringLength = hkString::strLen( timerName ) + 1;
			m_startCommand = HK_NULL;
			m_depth = 0;
		}

		hkMonitorStreamAnalyzer::Node			m_node;
		int										m_stringLength;	// including terminator. For memcmp() instead of strcmp().
		const hkMonitorStream::TimerCommand*	m_startCommand;
		int										m_depth;
	};
}

void HK_CALL hkMonitorStreamAnalyzer::gatherRawStreamTimerValues(
	const char* frameStart, const char* frameEnd, const hkMonitorStreamFrameInfo& frameInfo,
	const char** timerNames, int numTimers, hkReal* valuesOut )
{
	if( numTimers <= 0 )
	{
		return;
	}

	// Create some temporary storage to collect the timings
	hkArray<hkMonitorStreamAnalyzer_TimerCollector>::Temp collectors( numTimers );
	for( int i=0; i<numTimers; ++i )
	{
		collectors[i].init( timerNames[i] );
	}

	// Walk the stream
	const char* currentStreamPtr = frameStart;
	while( currentStreamPtr < frameEnd ) 
	{

		{
			const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
			if ( hkUlong(com->m_commandAndMonitor) < HK_JOB_TYPE_MAX )
			{
				int* elfId = (int*)com;
				currentStreamPtr = (char*)(elfId + 1);
				continue;
			}
		}

		const char* string = reinterpret_cast<const hkMonitorStream::Command*>(currentStreamPtr)->m_commandAndMonitor;
		switch( string[0] )
		{

		case 'T': // timer begin
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( collectors[i].m_startCommand )
					{
						collectors[i].m_depth++;
					}
					else if( hkString::memCmp( string + 2, collectors[i].m_node.m_name, collectors[i].m_stringLength ) == 0 )
					{
						collectors[i].m_startCommand = com;
						collectors[i].m_depth = 0;
					}
				}
				currentStreamPtr = (const char *)(com + 1);
				break;
			}

		case 'E': // timer end
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( collectors[i].m_startCommand )
					{
						if( collectors[i].m_depth == 0 )
						{
							collectors[i].m_node.setTimers( frameInfo, *collectors[i].m_startCommand, *com );
							collectors[i].m_startCommand = HK_NULL;
						}
						else
						{
							collectors[i].m_depth--;
						}
					}
				}
				currentStreamPtr = (const char *)(com + 1);
				break;
			}

		case 'L': // timer list begin
			{
				const hkMonitorStream::TimerBeginListCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginListCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( collectors[i].m_startCommand )
					{
						collectors[i].m_depth += 2;
					}
					else if( hkString::memCmp( string + 2, collectors[i].m_node.m_name, collectors[i].m_stringLength ) == 0 )
					{
						collectors[i].m_startCommand = com;
						collectors[i].m_depth = 1;
					}
					else if( hkString::memCmp( com->m_nameOfFirstSplit + 2, collectors[i].m_node.m_name, collectors[i].m_stringLength ) == 0 )
					{
						collectors[i].m_startCommand = com;
						collectors[i].m_depth = 0;
					}
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'S': // split list
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( collectors[i].m_startCommand )
					{
						if( collectors[i].m_depth == 0 )
						{
							collectors[i].m_node.setTimers( frameInfo, *collectors[i].m_startCommand, *com );
							collectors[i].m_startCommand = HK_NULL;
						}
					}
					else if( hkString::memCmp( string + 2, collectors[i].m_node.m_name, collectors[i].m_stringLength ) == 0 )
					{
						collectors[i].m_startCommand = com;
						collectors[i].m_depth = 0;
					}
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'l': // list end
			{
				const hkMonitorStream::TimerCommand* com = reinterpret_cast<const hkMonitorStream::TimerCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( collectors[i].m_startCommand )
					{
						if( collectors[i].m_depth <= 1 )
						{
							collectors[i].m_node.setTimers( frameInfo, *collectors[i].m_startCommand, *com );
							collectors[i].m_startCommand = HK_NULL;
						}
						collectors[i].m_depth -= 2;
					}
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'O': // object name
			{
				const hkMonitorStream::TimerBeginObjectNameCommand* com = reinterpret_cast<const hkMonitorStream::TimerBeginObjectNameCommand*>( currentStreamPtr );
				currentStreamPtr = (const char *)(com + 1);
				break;
			}

		case 'M':
			{
				const hkMonitorStream::AddValueCommand* com = reinterpret_cast<const hkMonitorStream::AddValueCommand*>( currentStreamPtr );
				for( int i=0; i<numTimers; ++i )
				{
					if( hkString::memCmp( string + 2, collectors[i].m_node.m_name, collectors[i].m_stringLength ) == 0 )
					{
						collectors[i].m_node.m_type = hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE;
						collectors[i].m_node.m_value[frameInfo.m_indexOfTimer0] = com->m_value;
						collectors[i].m_node.m_count[frameInfo.m_indexOfTimer0] += 1;
					}
				}
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'P':
		case 'p':
		case 'N': // nop
			{
				const hkMonitorStream::Command* com = reinterpret_cast<const hkMonitorStream::Command*>( currentStreamPtr );
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'm': // memory
			{
				const hkMonitorStream::MemoryCommand* com = reinterpret_cast<const hkMonitorStream::MemoryCommand*>( currentStreamPtr );
				currentStreamPtr = (const char*)(com + 1);
				break;
			}

		case 'F': // new frame
			{
				HK_ASSERT( 0x39129e52, 0 );
				break;
			}

		default:
			{
				HK_WARN( 0x3d7745e4, "Invalid monitor capture data" );
				break;
			}
		}
	}

	// Copy values out
	HK_ASSERT(0x1c538ca3, frameInfo.m_indexOfTimer0 == 0 );
	for( int i=0; i<numTimers; ++i )
	{
		valuesOut[i] = collectors[i].m_node.m_value[0];
	}
}

#endif // !SIM_SPU

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
