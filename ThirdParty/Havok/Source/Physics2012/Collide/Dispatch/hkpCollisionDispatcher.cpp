/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/String/hkStringBuf.h>

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpContactMgrFactory.h>
#include <Physics2012/Collide/BroadPhase/3AxisSweep/hkp3AxisSweep.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>
#include <Physics2012/Collide/Dispatch/Agent3Bridge/hkpAgent3Bridge.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>
#include <Physics2012/Collide/Filter/DefaultConvexList/hkpDefaultConvexListFilter.h>
#include <Physics2012/Collide/Filter/Null/hkpNullCollisionFilter.h>

#if defined(HK_PLATFORM_WIN32)
	HK_COMPILE_TIME_ASSERT( sizeof(hkpCollisionDispatcher::Agent3FuncsIntern) != sizeof(hkpCollisionDispatcher::Agent3Funcs) );
#endif

#if 0
static void HK_CALL hkNullGetPenetrationsFunc   (const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( A.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( B.getShape()->getType() ) );
	HK_WARN_ONCE(0x38206782,  "Have you called hkpAgentRegisterUtil::registerAllAgents? Do not know how to dispatch types " << typeA << " vs " << typeB);
}

static void HK_CALL hkNullGetClosestPointsFunc (const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpCdPointCollector& output )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( A.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( B.getShape()->getType() ) );
	HK_WARN_ONCE(0x19fb8b53,  "Have you called hkpAgentRegisterUtil::registerAllAgents? Do not know how to dispatch types " << typeA << " vs " << typeB);
}

static void HK_CALL hkNullLinearCastFunc       (const hkpCdBody& A, const hkpCdBody& B, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& castCollector, hkpCdPointCollector* startCollector )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( A.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( B.getShape()->getType() ) );
	HK_WARN_ONCE(0x523361cf,  "Have you called hkpAgentRegisterUtil::registerAllAgents? Do not know how to dispatch types " << typeA << " vs " << typeB);
}
#endif

namespace hkNullAgent3
{
	hkpAgentData* HK_CALL create( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* freeMemory )
	{
		HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( input.m_bodyA->getShape()->getType() ) );
		HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( input.m_bodyB->getShape()->getType() ) );
		HK_WARN_ONCE(0x523361df,  "Have you called hkpAgentRegisterUtil::registerAllAgents? Do not know how to dispatch types " << typeA << " vs " << typeB);
		entry->m_streamCommand = hkAgent3::STREAM_NULL;
		return freeMemory;
	}

	void HK_CALL destroy( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
	{

	}

	hkpAgentData*  HK_CALL process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormalOut, hkpProcessCollisionOutput& result)
	{
		return agentData;
	}
}



void hkpCollisionDispatcher::resetCreationFunctions()
{
	//
	// reset all response types
	//
	m_numAgent2Types = 1;
	m_numAgent3Types = 1;
	{
		for (int i = 0; i < HK_MAX_SHAPE_TYPE; ++i)
		{
			for (int j = 0; j < HK_MAX_SHAPE_TYPE; ++j)
			{
				if ( m_debugAgent2Table )
				{
					m_debugAgent2Table    [0][i][j].m_priority = 100;
					m_debugAgent2TablePred[0][i][j].m_priority = 100;
					m_debugAgent3Table    [0][i][j].m_priority = 100;
					m_debugAgent3TablePred[0][i][j].m_priority = 100;
				}
				m_agent2Types[i][j] = HK_AGENT_TYPE_NULL;
				m_agent3Types[i][j] = HK_AGENT_TYPE_NULL;
				m_agent2TypesPred[i][j] = HK_AGENT_TYPE_NULL;
				m_agent3TypesPred[i][j] = HK_AGENT_TYPE_NULL;
			}
		}
	}

	AgentFuncs& f2 = m_agent2Func[HK_AGENT_TYPE_NULL];
	f2.m_createFunc = hkpNullAgent::createNullAgent;
	f2.m_getPenetrationsFunc = hkpNullAgent::staticGetPenetrations;
	f2.m_getClosestPointFunc = hkpNullAgent::staticGetClosestPoints;
	f2.m_linearCastFunc = hkpNullAgent::staticLinearCast;
	f2.m_isFlipped = false;
	f2.m_isPredictive = true;


	Agent3FuncsIntern& f3 = m_agent3Func[HK_AGENT_TYPE_NULL];
	f3.m_createFunc   = hkNullAgent3::create;
	f3.m_destroyFunc  = hkNullAgent3::destroy;
	f3.m_cleanupFunc  = HK_NULL; //hkNullAgent3::cleanup;
	f3.m_removePointFunc  = HK_NULL; //hkNullAgent3::cleanup;
	f3.m_commitPotentialFunc  = HK_NULL; //hkNullAgent3::cleanup;
	f3.m_createZombieFunc  = HK_NULL; //hkNullAgent3::cleanup;
	f3.m_updateFilterFunc  = HK_NULL; //hkNullAgent3::cleanup;
	f3.m_sepNormalFunc = HK_NULL; //hkNullAgent3::sepNormal;
	f3.m_invalidateTimFunc = HK_NULL;
	f3.m_warpTimeFunc = HK_NULL;
	f3.m_processFunc  = hkNullAgent3::process;
	f3.m_symmetric    = hkAgent3::IS_SYMMETRIC;

	// register default agent silently
	HK_ON_DEBUG( int bridgeId = ) hkAgent3Bridge::registerAgent3(this);
	HK_ASSERT( 0xf05dae13, bridgeId == HK_AGENT_TYPE_BRIDGE );
	m_collisionAgentRegistered = false;

	// reset the priorities of the bridge
	if ( m_debugAgent3Table )
	{
		for (int i = 0; i < HK_MAX_SHAPE_TYPE; ++i)
		{
			for (int j = 0; j < HK_MAX_SHAPE_TYPE; ++j)
			{
				m_debugAgent3Table    [0][i][j].m_priority = 100;
				m_debugAgent3TablePred[0][i][j].m_priority = 100;
			}
		}
	}
}

hkpCollisionDispatcher::hkpCollisionDispatcher(CreateFunc defaultCollisionAgent, hkpContactMgrFactory* defaultContactMgrFactory )
:	m_defaultCollisionAgent(defaultCollisionAgent)
{
#ifdef HK_DEBUG
	
	// Those blocks need to be explicitly notified to the memory tracker since they are allocated in a special way using the debug allocator.
	// Since we are using the HK_MEMORY_TRACKER_NEW_RAW() macro, references inside the DebugEntry class will not be followed (this is fine since this class is a Pod type).
	// We are using the name buffer_hkpCollisionDispatcher::DebugTable so that those blocks will be collapsed in the hkpCollisionDispatcher block during inspection.
	m_debugAgent2Table = static_cast<hkpCollisionDispatcher::DebugTable*>(hkMemDebugBlockAlloc<void>(sizeof(hkpCollisionDispatcher::DebugTable)));
	HK_MEMORY_TRACKER_NEW_RAW("buffer_hkpCollisionDispatcher::DebugTable", m_debugAgent2Table, sizeof(hkpCollisionDispatcher::DebugTable));
	m_debugAgent2TablePred = static_cast<hkpCollisionDispatcher::DebugTable*>(hkMemDebugBlockAlloc<void>(sizeof(hkpCollisionDispatcher::DebugTable)));
	HK_MEMORY_TRACKER_NEW_RAW("buffer_hkpCollisionDispatcher::DebugTable", m_debugAgent2TablePred, sizeof(hkpCollisionDispatcher::DebugTable));
	m_debugAgent3Table = static_cast<hkpCollisionDispatcher::DebugTable*>(hkMemDebugBlockAlloc<void>(sizeof(hkpCollisionDispatcher::DebugTable)));
	HK_MEMORY_TRACKER_NEW_RAW("buffer_hkpCollisionDispatcher::DebugTable", m_debugAgent3Table, sizeof(hkpCollisionDispatcher::DebugTable));
	m_debugAgent3TablePred = static_cast<hkpCollisionDispatcher::DebugTable*>(hkMemDebugBlockAlloc<void>(sizeof(hkpCollisionDispatcher::DebugTable)));
	HK_MEMORY_TRACKER_NEW_RAW("buffer_hkpCollisionDispatcher::DebugTable", m_debugAgent3TablePred, sizeof(hkpCollisionDispatcher::DebugTable));
#else
	m_debugAgent2Table		= HK_NULL;
	m_debugAgent2TablePred  = HK_NULL;
	m_debugAgent3Table		= HK_NULL;
	m_debugAgent3TablePred	= HK_NULL;
#endif

	m_collisionAgentRegistered = false;
	m_checkEnabled = true;
	m_numAgent3Types = 0;
	m_midphaseAgent3Registered = false;

	//
	// reset all agents
	//
	{
		for ( int i = 0; i < HK_MAX_RESPONSE_TYPE; i++ )
		{
			for ( int j = 0; j < HK_MAX_RESPONSE_TYPE; j++ )
			{
				m_contactMgrFactory[i][j] = defaultContactMgrFactory;
				if ( defaultContactMgrFactory )
				{
					defaultContactMgrFactory->addReference();
				}
			}
		}
	}


	//
	//	reset all shape rules
	//
	{
		for (int i = 0; i < HK_MAX_SHAPE_TYPE; i++ )
		{
			m_hasAlternateType[i] = 1<<i;
		}
	}

	resetCreationFunctions();


	// we know that our bridge agent is very small, so need to
	// reserve lots of memory for 32 bytes
	m_agent3Registered = false;
}

void hkpCollisionDispatcher::setEnableChecks( hkBool checkEnabled)
{
	m_checkEnabled = checkEnabled;
}

void hkpCollisionDispatcher::disableDebugging()
{
	if ( m_debugAgent2Table )
	{
		
		hkMemDebugBlockFree<void>(m_debugAgent2Table, sizeof(hkpCollisionDispatcher::DebugTable));
		HK_MEMORY_TRACKER_DELETE_RAW(m_debugAgent2Table);
		hkMemDebugBlockFree<void>(m_debugAgent2TablePred, sizeof(hkpCollisionDispatcher::DebugTable));
		HK_MEMORY_TRACKER_DELETE_RAW(m_debugAgent2TablePred);
		hkMemDebugBlockFree<void>(m_debugAgent3Table, sizeof(hkpCollisionDispatcher::DebugTable));
		HK_MEMORY_TRACKER_DELETE_RAW(m_debugAgent3Table);
		hkMemDebugBlockFree<void>(m_debugAgent3TablePred, sizeof(hkpCollisionDispatcher::DebugTable));
		HK_MEMORY_TRACKER_DELETE_RAW(m_debugAgent3TablePred);
		m_debugAgent2Table		= HK_NULL;
		m_debugAgent2TablePred  = HK_NULL;
		m_debugAgent3Table		= HK_NULL;
		m_debugAgent3TablePred	= HK_NULL;
	}
}

hkpCollisionDispatcher::~hkpCollisionDispatcher()
{
	disableDebugging();

	{
		for ( int i = 0; i < HK_MAX_RESPONSE_TYPE; i++ )
		{
			for ( int j = 0; j < HK_MAX_RESPONSE_TYPE; j++ )
			{
				if (m_contactMgrFactory[i][j])
				{
					m_contactMgrFactory[i][j]->removeReference();
				}
			}
		}
	}
}

void hkpCollisionDispatcher::registerCollisionAgent(AgentFuncs& f, hkpShapeType typeA, hkpShapeType typeB)
{
	HK_ASSERT2( 0xad000301, m_numAgent2Types < HK_MAX_AGENT2_TYPES, "You are running out of agent2 entries");


	//
	//	Register tables
	//
	m_agent2Func[ m_numAgent2Types ] = f;

	internalRegisterCollisionAgent( m_agent3Types, HK_AGENT_TYPE_BRIDGE, typeA, typeB, typeA, typeB, m_debugAgent3Table, 0 );
	internalRegisterCollisionAgent( m_agent2Types, m_numAgent2Types,     typeA, typeB, typeA, typeB, m_debugAgent2Table, 0 );

	if ( f.m_isPredictive )
	{
		internalRegisterCollisionAgent( m_agent3TypesPred, HK_AGENT_TYPE_BRIDGE, typeA, typeB, typeA, typeB, m_debugAgent3TablePred, 0 );
		internalRegisterCollisionAgent( m_agent2TypesPred, m_numAgent2Types,     typeA, typeB, typeA, typeB, m_debugAgent2TablePred, 0 );
	}
	m_numAgent2Types++;
}

void hkpCollisionDispatcher::registerCollisionAgent2(AgentFuncs& f, hkpShapeType typeA, hkpShapeType typeB)
{
	HK_ASSERT2( 0xad000301, m_numAgent2Types < HK_MAX_AGENT2_TYPES, "You are running out of agent2 entries");

	//
	//	Register tables
	//
	m_agent2Func[ m_numAgent2Types ] = f;

	internalRegisterCollisionAgent( m_agent2Types, m_numAgent2Types, typeA, typeB, typeA, typeB, m_debugAgent2Table, 0 );

	if ( f.m_isPredictive )
	{
		internalRegisterCollisionAgent( m_agent2TypesPred, m_numAgent2Types, typeA, typeB, typeA, typeB, m_debugAgent2TablePred, 0 );
	}

	m_numAgent2Types++;
}

int hkpCollisionDispatcher::registerAgent3( Agent3Funcs& f, hkpShapeType typeA, hkpShapeType typeB )
{
	m_agent3Registered = true;

	HK_ASSERT2( 0xf0180404, m_numAgent3Types < HK_MAX_AGENT3_TYPES, "You are running out of agent3 entries");

	//
	//	check for symmetric
	//
	Agent3FuncsIntern f3;
	(Agent3Funcs&)f3 = f;
	f3.m_symmetric = hkAgent3::IS_SYMMETRIC;

	if ( (typeA != typeB) && (f.m_ignoreSymmetricVersion == false) )
	{
		HK_ASSERT(0xf0342354, f.m_reusePreviousEntry == false );
		f3.m_symmetric = hkAgent3::IS_NOT_SYMMETRIC_AND_FLIPPED;
		m_agent3Func[ m_numAgent3Types ] = f3;
		internalRegisterCollisionAgent( m_agent3Types, m_numAgent3Types, typeB, typeA, typeB, typeA, m_debugAgent3Table, 0 );

		if ( f3.m_isPredictive )
		{
			internalRegisterCollisionAgent( m_agent3TypesPred, m_numAgent3Types, typeB, typeA, typeB, typeA, m_debugAgent3TablePred, 0 );
		}
		m_numAgent3Types++;
		HK_ASSERT2( 0xf0180404, m_numAgent3Types < HK_MAX_AGENT3_TYPES, "You are running out of agent3 entries");

		f3.m_symmetric = hkAgent3::IS_NOT_SYMMETRIC;
	}

	int currentType;
	if ( !f.m_reusePreviousEntry )
	{
		currentType = m_numAgent3Types++;
		m_agent3Func[ currentType ] = f3;
	}
	else
	{
		currentType = m_numAgent3Types-1;
		HK_ON_DEBUG( Agent3FuncsIntern& fd = m_agent3Func[ currentType ] );
		HK_ASSERT2( 0xf0653de9, fd.m_processFunc == f3.m_processFunc && fd.m_createFunc == f3.m_createFunc && fd.m_isPredictive == f3.m_isPredictive, "You have to use the same functions again" );
	}

	internalRegisterCollisionAgent( m_agent3Types, currentType, typeA, typeB, typeA, typeB, m_debugAgent3Table, 0 );

	if ( f3.m_isPredictive )
	{
		internalRegisterCollisionAgent( m_agent3TypesPred, currentType, typeA, typeB, typeA, typeB, m_debugAgent3TablePred, 0 );
	}

	return currentType;
}


void hkpCollisionDispatcher::internalRegisterCollisionAgent(hkUchar agentTypesTable[HK_MAX_SHAPE_TYPE][HK_MAX_SHAPE_TYPE], int agentType, hkpShapeType typeA, hkpShapeType typeB, hkpShapeType origA, hkpShapeType origB, DebugTable *debugTable, int depth)
{
	HK_ASSERT3(0x53270a5c,  hkcdShapeType::ALL_SHAPE_TYPES <= typeA && typeA < (hkpShapeType)HK_MAX_SHAPE_TYPE, "You can only access types between [hkcdShapeType::ALL_SHAPE_TYPES .." << HK_MAX_SHAPE_TYPE << "]");
	HK_ASSERT3(0x328325a1,  hkcdShapeType::ALL_SHAPE_TYPES <= typeB && typeB < (hkpShapeType)HK_MAX_SHAPE_TYPE, "You can only access types between [hkcdShapeType::ALL_SHAPE_TYPES .." << HK_MAX_SHAPE_TYPE << "]");
	HK_ASSERT2(0x76e7dd75,  depth < 10, "Infinite loop: your alternate shape types have a circular dependency");
	m_collisionAgentRegistered = true;


	//
	//	Traverse the hierarchy to more specialized agents
	//  If there is a rule:   shapeA inheritsFrom shapeB    and shapeB is either origA or origB, call this function with shapeA
	//
	{
		for ( int i = 0; i < m_shapeInheritance.getSize(); i++ )
		{
			ShapeInheritance& si = m_shapeInheritance[i];

			if ( si.m_alternateType == typeA  )
			{
				internalRegisterCollisionAgent( agentTypesTable, agentType, si.m_primaryType, typeB, origA, origB, debugTable, depth+1 );
			}
			if ( si.m_alternateType == typeB )
			{
				internalRegisterCollisionAgent( agentTypesTable, agentType, typeA, si.m_primaryType, origA, origB, debugTable, depth+1 );
			}
		}
	}

	{
		//
		//	Some helper code to replace  hkcdShapeType::ALL_SHAPE_TYPES by an iteration
		//
		int beginA = typeA;
		int beginB = typeB;
		int endA = typeA+1;
		int endB = typeB+1;
		int priority = depth;
		if ( typeA == hkcdShapeType::ALL_SHAPE_TYPES )
		{
			beginA = hkcdShapeType::FIRST_SHAPE_TYPE;
			endA = HK_MAX_SHAPE_TYPE;
			priority++;
		}
		if ( typeB == hkcdShapeType::ALL_SHAPE_TYPES )
		{
			beginB = hkcdShapeType::FIRST_SHAPE_TYPE;
			endB = HK_MAX_SHAPE_TYPE;
			priority++;
		}

		//
		//	Iterate over all combinations (if you are not using hkcdShapeType::ALL_SHAPE_TYPES, than this results in a single iteration)
		//
		for (int a = beginA; a < endA; a++)
		{
			for (int b = beginB; b < endB; b++)
			{
				//
				//	Fill in the agent table
				//
				agentTypesTable[a][b] = (hkUchar)agentType;

				//
				//	Do some debugging
				//
				if ( debugTable )
				{
					DebugEntry& de = debugTable[0][a][b];

					if ( m_checkEnabled && priority > de.m_priority )
					{
						//
						//	error
						//
						const char* oldA = hkGetShapeTypeName( hkpShapeType(de.m_typeA) );
						const char* oldB = hkGetShapeTypeName( hkpShapeType(de.m_typeB) );
						const char* newA = hkGetShapeTypeName( hkpShapeType(typeA) );
						const char* newB = hkGetShapeTypeName( hkpShapeType(typeB) );
						char buffer[1024];
						hkString::snprintf( buffer, 1000,
							"Agent handling types <%s-%s> would override more specialized agent <%s-%s>\n"
							"Maybe the order of registering your collision agent is wrong, make sure you register your alternate type agents first",
							newA,newB, oldA,oldB );
						HK_ASSERT2(0x62b50e8a,  0, buffer );

					}
					HK_ASSERT2(0x4271c5c2,  priority < 256 && origA < 256 && origB <256, "Currently there is a limitation of 256 shape types" );

					de.m_priority = char(priority);
					de.m_typeA = char(origA);
					de.m_typeB = char(origB);
				}
			}
		}
	}
}



// subfunction which keeps the m_hasAlternateType up to date
void hkpCollisionDispatcher::updateHasAlternateType( hkpShapeType primaryType, hkpShapeType alternateType, int depth )
{
	HK_ASSERT2(0x33af9996,  depth < 100, "Your shape dependency graph contains a cycle");

		// add all the children
	m_hasAlternateType[ primaryType ] = m_hasAlternateType[ primaryType ] | m_hasAlternateType[ alternateType ];


	//	Traverse up the hierarchy
	{
		for (int i = 0; i < m_shapeInheritance.getSize(); i++)
		{
			ShapeInheritance& si = m_shapeInheritance[i];
			if ( si.m_alternateType == primaryType  )
			{
				// recurse up
				updateHasAlternateType( si.m_primaryType, si.m_alternateType, depth+1 );
			}
		}
	}
}

void hkpCollisionDispatcher::registerAlternateShapeType( hkpShapeType primaryType, hkpShapeType alternateType )
{
	HK_ASSERT3(0x3b2d0f10,  hkcdShapeType::FIRST_SHAPE_TYPE <= primaryType   && primaryType   < (hkpShapeType)HK_MAX_SHAPE_TYPE, "You can only access types between [HK_FIRST_SHAPE_TYPE ..." << HK_MAX_SHAPE_TYPE << "]");
	HK_ASSERT3(0x549a2997,  hkcdShapeType::FIRST_SHAPE_TYPE <= alternateType && alternateType < (hkpShapeType)HK_MAX_SHAPE_TYPE, "You can only access types between [HK_FIRST_SHAPE_TYPE ..." << HK_MAX_SHAPE_TYPE << "]");
	HK_ASSERT2(0x350560c3,  primaryType != alternateType, "Your primary type must be different from the alternateType" );

	//
	//	If we already have registered agents, we have to unregister them all,
	//  register our shape type and reregister them all again
	//
	if ( m_collisionAgentRegistered != false )
	{
		HK_ASSERT2(0x70111344, 0,  "You have to register all shapeTypes before call registerCollisionAgent() ");
	}

	//
	//	Search for duplicated entries
	//
	{
		for (int i = 0; i < m_shapeInheritance.getSize(); i++)
		{
			ShapeInheritance& si = m_shapeInheritance[i];
			if ( si .m_primaryType == primaryType && si.m_alternateType == alternateType )
			{
				HK_WARN(0x3e3a6c67, "Agent registered twice, deleting the original entry");
				m_shapeInheritance.removeAtAndCopy(i);
				i--;
			}
		}
	}

	//
	//	updateHasAlternateType
	//
	{
		updateHasAlternateType( primaryType, alternateType, 0 );
	}


	//
	//	Add to our list
	//
	{
		ShapeInheritance& si = m_shapeInheritance.expandOne();
		si.m_primaryType = primaryType;
		si.m_alternateType = alternateType;
	}
}



void hkpCollisionDispatcher::registerContactMgrFactory( hkpContactMgrFactory* fac, int responseA, int responseB )
{
	HK_ASSERT3(0x14b1ad5b,  0 <= responseA && responseA < HK_MAX_RESPONSE_TYPE, "Response Type A is outside [ 0 .. " << HK_MAX_RESPONSE_TYPE << "]" );
	HK_ASSERT3(0x154b139d,  0 <= responseB && responseB < HK_MAX_RESPONSE_TYPE, "Response Type B is outside [ 0 .. " << HK_MAX_RESPONSE_TYPE << "]" );

	fac->addReference();
	m_contactMgrFactory[ responseB] [responseA ]->removeReference();
	m_contactMgrFactory[ responseB] [responseA ] = fac;

	fac->addReference();
	m_contactMgrFactory[ responseA ][ responseB ]->removeReference();
	m_contactMgrFactory[ responseA ][ responseB ] = fac;
}

void hkpCollisionDispatcher::registerContactMgrFactoryWithAll( hkpContactMgrFactory* fac, int responseA )
{
	HK_ASSERT3(0x478d2d55,  responseA < HK_MAX_RESPONSE_TYPE, "You can only register types between [0.." << HK_MAX_RESPONSE_TYPE << "]");

	for (int i = 0; i < HK_MAX_RESPONSE_TYPE; i++ )
	{
		fac->addReference();

		m_contactMgrFactory[ i ][ responseA ]->removeReference();
		m_contactMgrFactory[ i ][ responseA ] = fac;

		fac->addReference();

		m_contactMgrFactory[ responseA ][ i ]->removeReference();
		m_contactMgrFactory[ responseA ][ i ] = fac;
	}
}

void hkpCollisionDispatcher::getClosestPoints( const hkpShape* shapeA, const hkTransform& transformA, const hkpShape* shapeB, const hkTransform& transformB, hkReal collisionTolerance, hkpCdPointCollector& collector )
{
	hkpNullCollisionFilter filter;
	hkpDefaultConvexListFilter defaultConvexFilter;
	hkpCollisionInput collisionInput;
	{
		collisionInput.m_dispatcher       = this;
		collisionInput.m_filter           = &filter;
		collisionInput.m_convexListFilter = &defaultConvexFilter;
		collisionInput.setTolerance( collisionTolerance );
	}
	hkpCollidable collA( shapeA, &transformA );
	hkpCollidable collB( shapeB, &transformB );

	hkpCollisionDispatcher::GetClosestPointsFunc closestPointsFunc = getGetClosestPointsFunc(shapeA->getType(), shapeB->getType());
	HK_ASSERT3( 0xf0fd45ed, closestPointsFunc, "No collision agent registered between types " << hkGetShapeTypeName( shapeA->getType() ) << " and " << hkGetShapeTypeName( shapeB->getType() ) );

	closestPointsFunc(collA, collB, collisionInput, collector);
}

void hkpCollisionDispatcher::getPenetrations( const hkpShape* shapeA, const hkTransform& transformA, const hkpShape* shapeB, const hkTransform& transformB, hkReal collisionTolerance, hkpCdBodyPairCollector& collector )
{
	hkpNullCollisionFilter filter;
	hkpDefaultConvexListFilter defaultConvexFilter;
	hkpCollisionInput collisionInput;
	{
		collisionInput.m_dispatcher       = this;
		collisionInput.m_filter           = &filter;
		collisionInput.m_convexListFilter = &defaultConvexFilter;
		collisionInput.setTolerance( collisionTolerance );
	}
	hkpCollidable collA( shapeA, &transformA );
	hkpCollidable collB( shapeB, &transformB );

	hkpCollisionDispatcher::GetPenetrationsFunc penetrationsFunc = getGetPenetrationsFunc(shapeA->getType(), shapeB->getType());
	HK_ASSERT3( 0xf0fd45ea, penetrationsFunc, "No collision agent registered between types " << hkGetShapeTypeName( shapeA->getType() ) << " and " << hkGetShapeTypeName( shapeB->getType() ) );

	penetrationsFunc(collA, collB, collisionInput, collector);
}

void hkpCollisionDispatcher::debugPrintTable()
{
	HK_REPORT_SECTION_BEGIN(0x5e4345e4, "hkpCollisionDispatcher::debugPrintTable" );

	if ( !m_debugAgent2Table || !m_debugAgent2TablePred)
	{
		HK_WARN( 0xf0324455, "Debugging disabled, cannot print debug table" );
		return;
	}
	char buf[255];
	{
		for (hkpShapeType a = hkpShapeType(0); a < (hkpShapeType)HK_MAX_SHAPE_TYPE; a = hkpShapeType(a+1))
		{
			hkStringBuf str("\nEntries for (continuous)", hkGetShapeTypeName(hkpShapeType(a)) );
			HK_REPORT(str);

			for (int b = hkcdShapeType::FIRST_SHAPE_TYPE; b < HK_MAX_SHAPE_TYPE; b++)
			{
				DebugEntry& de = m_debugAgent2TablePred[0][a][b];
				if ( de.m_priority >= 100 )
				{
					continue;
				}

				const char* sB = hkGetShapeTypeName( hkpShapeType(b) );
				const char* oldA = hkGetShapeTypeName( hkpShapeType(de.m_typeA) );
				const char* oldB = hkGetShapeTypeName( hkpShapeType(de.m_typeB) );

				hkString::snprintf(buf, 255, "vs %30s <%i:%s-%s>", sB, de.m_priority, oldA, oldB );
				HK_REPORT(buf);
			}
		}
	}
	{
		for (int a = 0; a < HK_MAX_SHAPE_TYPE; a++)
		{
			hkStringBuf str("\nEntries for (discrete)", hkGetShapeTypeName( hkpShapeType(a) ) );
			HK_REPORT(str);

			for (int b = hkcdShapeType::FIRST_SHAPE_TYPE; b < HK_MAX_SHAPE_TYPE; b++)
			{
				DebugEntry& de = m_debugAgent2Table[0][a][b];
				if ( de.m_priority >= 100 )
				{
					continue;
				}

				const char* sB = hkGetShapeTypeName( hkpShapeType(b) );
				const char* oldA = hkGetShapeTypeName( hkpShapeType(de.m_typeA) );
				const char* oldB = hkGetShapeTypeName( hkpShapeType(de.m_typeB) );

				hkString::snprintf(buf, 255, "vs %30s <%i:%s-%s>", sB, de.m_priority, oldA, oldB );
				HK_REPORT(buf);
			}
		}
	}
	HK_REPORT_SECTION_END();
}

void hkpCollisionDispatcher::initCollisionQualityInfo( InitCollisionQualityInfo& input )
{
	m_expectedMinPsiDeltaTime = input.m_minDeltaTime;
	m_expectedMaxLinearVelocity = input.m_maxLinearVelocity;

	// This is the distance an objects travels under gravity within one timestep
	float distPerTimeStep = 0.5f * float(input.m_gravityLength * input.m_minDeltaTime * input.m_minDeltaTime);

	{
		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];

		sq.m_enableToiWeldRejection =  input.m_enableToiWeldRejection;

		sq.m_keepContact         = input.m_collisionTolerance;
		sq.m_create4dContact     = input.m_collisionTolerance;

		sq.m_manifoldTimDistance = input.m_collisionTolerance;
		if ( input.m_enableNegativeManifoldTims )
		{
			sq.m_manifoldTimDistance = -2.0f * distPerTimeStep;
		}
		sq.m_createContact       = input.m_collisionTolerance;
		if ( input.m_enableNegativeToleranceToCreateNon4dContacts )
		{
			sq.m_createContact       = -1.0f * distPerTimeStep;
		}

		sq.m_maxContraintViolation    = HK_REAL_MAX;
		sq.m_useContinuousPhysics	  = false;
		sq.m_useSimpleToiHandling     = false;
		sq.m_constraintPriority       = input.m_defaultConstraintPriority;

		hkReal d = -1e20f;
		sq.m_minSeparation		= d * 0.5f;
		sq.m_minExtraSeparation	= d * 0.5f;
		sq.m_toiSeparation		= d * 0.1f;
		sq.m_toiExtraSeparation	= d * 0.1f;
		sq.m_toiAccuracy		= hkMath::fabs(d * 0.05f);
		sq.m_minSafeDeltaTime	= 1.0f;
		sq.m_minAbsoluteSafeDeltaTime = 1.0f;
		sq.m_minToiDeltaTime	= 1.0f;
	}

	// copy the PSI-quality-info values as defaults for further quality infos
	{
		m_collisionQualityInfo[ COLLISION_QUALITY_TMP_EXPAND_MANIFOLD ] = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];
		m_collisionQualityInfo[ COLLISION_QUALITY_SIMPLIFIED_TOI ]      = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];
		m_collisionQualityInfo[ COLLISION_QUALITY_TOI ]                 = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];
		m_collisionQualityInfo[ COLLISION_QUALITY_TOI_HIGHER ]          = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];
		m_collisionQualityInfo[ COLLISION_QUALITY_TOI_FORCED ]          = m_collisionQualityInfo[ COLLISION_QUALITY_PSI ];
	}

	{
		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_TMP_EXPAND_MANIFOLD ];
		sq.m_manifoldTimDistance = input.m_collisionTolerance;
		sq.m_create4dContact     = input.m_collisionTolerance;
		sq.m_createContact       = input.m_collisionTolerance;
	}

	// This is the smallest thickness of the object diveded by the allowedPenetration
	const hkReal radiusToAllowedPenetrationRatio = 5.0f; // This increases collisionAgent's tolerance.m_safeDeltaTimeStep

	{

		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_SIMPLIFIED_TOI ];

		if (input.m_wantContinuousCollisionDetection)
		{
			// The maximum relative penetration distance allowed for the first (initial) TOI. The remaining penetration depth (until the full
			// penetration depth is reached) will be subdivided into a customizable number of TOIs.
			const hkReal maxInitialRelPenetration = 0.5f;

			sq.m_minSeparation		= -1.0f * maxInitialRelPenetration;
			sq.m_minExtraSeparation	= -1.0f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationSimplifiedToi-1);
			sq.m_toiSeparation		= -0.5f * maxInitialRelPenetration;
			sq.m_toiExtraSeparation	= -0.7f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationSimplifiedToi-1);
			sq.m_toiAccuracy		= -0.3f * sq.m_toiExtraSeparation;
			sq.m_maxContraintViolation  = -1.0f * sq.m_minExtraSeparation;

			sq.m_minSafeDeltaTime   = radiusToAllowedPenetrationRatio / input.m_maxLinearVelocity;
			sq.m_minAbsoluteSafeDeltaTime   = 0.005f / input.m_maxLinearVelocity; // Should work for bodies 1cm in size.
			sq.m_minToiDeltaTime	= -2.0f * sq.m_minExtraSeparation / input.m_maxLinearVelocity;

			sq.m_constraintPriority = input.m_toiConstraintPriority;
			sq.m_useContinuousPhysics   = true;
			sq.m_useSimpleToiHandling   = true;

			HK_ASSERT2(0xad2342da, sq.m_minToiDeltaTime > 0.0f && sq.m_minSafeDeltaTime > 0.0f, "Internal error: incorrect initialization of surface qualities.");
		}
	}

	{

		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_TOI ];

		if (input.m_wantContinuousCollisionDetection)
		{
			// The maximum relative penetration distance allowed for the first (initial) TOI. The remaining penetration depth (until the full
			// penetration depth is reached) will be subdivided into a customizable number of TOIs.
			const hkReal maxInitialRelPenetration = 0.5f;

			sq.m_minSeparation		= -1.0f * maxInitialRelPenetration;
			sq.m_minExtraSeparation	= -1.0f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToi-1);
			sq.m_toiSeparation		= -0.5f * maxInitialRelPenetration;
			sq.m_toiExtraSeparation	= -0.7f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToi-1);
			sq.m_toiAccuracy		= -0.3f * sq.m_toiExtraSeparation;
			sq.m_maxContraintViolation  = -1.0f * sq.m_minExtraSeparation;

			sq.m_minSafeDeltaTime   = radiusToAllowedPenetrationRatio / input.m_maxLinearVelocity;
			sq.m_minAbsoluteSafeDeltaTime   = 0.005f / input.m_maxLinearVelocity; // Should work for bodies 1cm in size.
			sq.m_minToiDeltaTime	= -2.0f * sq.m_minExtraSeparation / input.m_maxLinearVelocity;

			sq.m_constraintPriority = input.m_toiConstraintPriority;
			sq.m_useContinuousPhysics   = true;

			HK_ASSERT2(0xad2342da, sq.m_minToiDeltaTime > 0.0f && sq.m_minSafeDeltaTime > 0.0f, "Internal error: incorrect initialization of surface qualities.");
		}
	}

	{
		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_TOI_HIGHER ];

		if (input.m_wantContinuousCollisionDetection)
		{
			// The maximum relative penetration distance allowed for the first (initial) TOI. The remaining penetration depth (until the full
			// penetration depth is reached) will be subdivided into a customizable number of TOIs.
			const hkReal maxInitialRelPenetration = 0.5f;

			sq.m_minSeparation		= -1.0f * maxInitialRelPenetration;
			sq.m_minExtraSeparation	= -1.0f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToiHigher-1);
			sq.m_toiSeparation		= -0.5f * maxInitialRelPenetration;
			sq.m_toiExtraSeparation	= -0.7f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToiHigher-1);
			sq.m_toiAccuracy		= -0.3f * sq.m_toiExtraSeparation;
			sq.m_maxContraintViolation  = -1.0f * sq.m_minExtraSeparation;

			sq.m_minSafeDeltaTime   = radiusToAllowedPenetrationRatio / input.m_maxLinearVelocity;
			sq.m_minAbsoluteSafeDeltaTime   = 0.005f / input.m_maxLinearVelocity; // Should work for bodies 1cm in size.
			sq.m_minToiDeltaTime	= -1.0f * sq.m_minExtraSeparation / input.m_maxLinearVelocity;

			sq.m_constraintPriority = input.m_toiHigherConstraintPriority;
			sq.m_useContinuousPhysics   = true;

			HK_ASSERT2(0xad2342da, sq.m_minToiDeltaTime > 0.0f && sq.m_minSafeDeltaTime > 0.0f, "Internal error: incorrect initialization of surface qualities.");
		}
	}

	{

		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_TOI_FORCED ];

		sq.m_keepContact         = input.m_collisionTolerance;

		if (input.m_wantContinuousCollisionDetection)
		{
			// The maximum relative penetration distance allowed for the first (initial) TOI. The remaining penetration depth (until the full
			// penetration depth is reached) will be subdivided into a customizable number of TOIs.
			const hkReal maxInitialRelPenetration = 0.4f;

			sq.m_minSeparation		= -1.0f * maxInitialRelPenetration;
			sq.m_minExtraSeparation	= -1.0f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToiForced-1);
			sq.m_toiSeparation		= -0.5f * maxInitialRelPenetration;
			sq.m_toiExtraSeparation	= -0.7f * (1.0f - maxInitialRelPenetration) / (input.m_numToisTillAllowedPenetrationToiForced-1);
			sq.m_toiAccuracy		= -0.3f * sq.m_toiExtraSeparation;
			sq.m_maxContraintViolation  = -1.0f * sq.m_minExtraSeparation;

			sq.m_minSafeDeltaTime   = radiusToAllowedPenetrationRatio / input.m_maxLinearVelocity;
			sq.m_minAbsoluteSafeDeltaTime   = 0.005f / input.m_maxLinearVelocity; // Should work for bodies 1cm in size.
			sq.m_minToiDeltaTime	= -1.0f * sq.m_minExtraSeparation / input.m_maxLinearVelocity;

			sq.m_constraintPriority = input.m_toiForcedConstraintPriority;
			sq.m_useContinuousPhysics   = true;

			HK_ASSERT2(0xad2342da, sq.m_minToiDeltaTime > 0.0f && sq.m_minSafeDeltaTime > 0.0f, "Internal error: incorrect initialization of surface qualities.");
		}
	}

	{
		hkpCollisionQualityInfo& sq = m_collisionQualityInfo[ COLLISION_QUALITY_CHARACTER ];
		sq = m_collisionQualityInfo[ COLLISION_QUALITY_TOI_FORCED ];

		// Set these values to the hkWorldCinfo::CONTACT_POINT_ACCEPT_ALWAYS defaults
		sq.m_manifoldTimDistance = input.m_collisionTolerance;
		sq.m_createContact       = input.m_collisionTolerance;
		sq.m_createContact       = 0.001f; 		
		sq.m_create4dContact     = 0.01f;
	}

	{
		for ( int i = 0; i < HK_COLLIDABLE_QUALITY_MAX; i++)
		{
			for ( int j = 0; j < HK_COLLIDABLE_QUALITY_MAX; j++)
			{
				m_collisionQualityTable[i][j] = COLLISION_QUALITY_PSI;
			}
		}
	}

#define T m_collisionQualityTable

#define L HK_COLLIDABLE_QUALITY_FIXED
#define K HK_COLLIDABLE_QUALITY_KEYFRAMED
#define R HK_COLLIDABLE_QUALITY_KEYFRAMED_REPORTING
#define D HK_COLLIDABLE_QUALITY_DEBRIS
#define S HK_COLLIDABLE_QUALITY_DEBRIS_SIMPLE_TOI
#define M HK_COLLIDABLE_QUALITY_MOVING
#define C HK_COLLIDABLE_QUALITY_CRITICAL
#define B HK_COLLIDABLE_QUALITY_BULLET
#define P HK_COLLIDABLE_QUALITY_CHARACTER

#define QI hkUchar(COLLISION_QUALITY_INVALID)
#define QP hkUchar(COLLISION_QUALITY_PSI)
#define QS hkUchar(COLLISION_QUALITY_SIMPLIFIED_TOI)
#define QT hkUchar(COLLISION_QUALITY_TOI)
#define QH hkUchar(COLLISION_QUALITY_TOI_HIGHER)
#define QF hkUchar(COLLISION_QUALITY_TOI_FORCED)
#define QC hkUchar(COLLISION_QUALITY_CHARACTER)


	T[L][L] = QI;	T[L][K] = QI;	T[L][R] = QT;	T[L][D] = QP;	T[L][S] = QS;	T[L][M] = QH;	T[L][C] = QF;	T[L][B] = QH; T[L][P] = QC;
	T[K][L] = QI;	T[K][K] = QI;	T[K][R] = QT;	T[K][D] = QP;	T[K][S] = QP;	T[K][M] = QT;	T[K][C] = QT;	T[K][B] = QT; T[K][P] = QT;
	T[R][L] = QT;	T[R][K] = QT;	T[R][R] = QT;	T[R][D] = QP;	T[R][S] = QP;	T[R][M] = QT;	T[R][C] = QT;	T[R][B] = QT; T[R][P] = QT;
	T[D][L] = QP;	T[D][K] = QP;	T[D][R] = QP;	T[D][D] = QP;	T[D][S] = QP;	T[D][M] = QP;	T[D][C] = QP;	T[D][B] = QT; T[D][P] = QP;
	T[S][L] = QS;	T[S][K] = QP;	T[S][R] = QP;	T[S][D] = QP;	T[S][S] = QP;	T[S][M] = QP;	T[S][C] = QP;	T[S][B] = QT; T[S][P] = QP;
	T[M][L] = QH;	T[M][K] = QT;	T[M][R] = QT;	T[M][D] = QP;	T[M][S] = QP;	T[M][M] = QP;	T[M][C] = QT;	T[M][B] = QT; T[M][P] = QT;
	T[C][L] = QF;	T[C][K] = QT;	T[C][R] = QT;	T[C][D] = QP;	T[C][S] = QP;	T[C][M] = QT;	T[C][C] = QT;	T[C][B] = QT; T[C][P] = QT;
	T[B][L] = QH;	T[B][K] = QT;	T[B][R] = QT;	T[B][D] = QT;	T[B][S] = QT;	T[B][M] = QT;	T[B][C] = QT;	T[B][B] = QT; T[B][P] = QT;
	T[P][L] = QC;	T[P][K] = QT;	T[P][R] = QT;	T[P][D] = QP;	T[P][S] = QP;	T[P][M] = QT;	T[P][C] = QT;	T[P][B] = QT; T[P][P] = QT;

}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
