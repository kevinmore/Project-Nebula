/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>

#include <Physics2012/Dynamics/World/Util/hkpWorldAgentUtil.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>

#define SIMPLE( ENUM_TYPE, STRUCT_TYPE, EXEC )								\
		case hkpPhysicsCommand::ENUM_TYPE:									\
		{																	\
			STRUCT_TYPE* com = static_cast<STRUCT_TYPE*>(begin);			\
			EXEC;															\
			const int STRUCT_SIZE = HK_NEXT_MULTIPLE_OF( 16, sizeof( STRUCT_TYPE ) );\
			begin = hkAddByteOffset( begin, STRUCT_SIZE );							\
			break;															\
		}

static void addConstraintToCriticalLockedIsland( hkpWorld* world, hkpConstraintInstance* constraint, int callbackRequest)
{
	hkpWorldOperationUtil::addConstraintToCriticalLockedIsland( world, constraint );
	constraint->m_internal->m_callbackRequest |= callbackRequest; 
}

void HK_CALL hkPhysicsCommandMachineProcess( hkpWorld* world, hkpPhysicsCommand* begin, int size )
{
	hkpPhysicsCommand* end = hkAddByteOffset(begin, size);
	while ( begin < end )
	{
		switch( begin->m_type )
		{
			SIMPLE( TYPE_ADD_CONSTRAINT_TO_LOCKED_ISLAND,      hkpAddConstraintToCriticalLockedIslandPhysicsCommand,      addConstraintToCriticalLockedIsland( world, com->m_object0, com->m_object1) );
			SIMPLE( TYPE_REMOVE_CONSTRAINT_FROM_LOCKED_ISLAND, hkpRemoveConstraintFromCriticalLockedIslandPhysicsCommand, hkpWorldOperationUtil::removeConstraintFromCriticalLockedIsland( world, com->m_object) );

			default:
				HK_ASSERT2( 0xf02ddd12, false, "Unknown hkpPhysicsCommand type");
				return;
		}
	}
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
