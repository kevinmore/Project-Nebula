/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintCallbackUtil.h>

static inline void HK_CALL hkpConstraintCallbackUtil_cleanupNullPointers( hkSmallArray<void*>& cleanupArray )
{
	for (int i = cleanupArray.getSize() - 1; i >= 0; i-- )
	{
		if ( cleanupArray[i] == HK_NULL )
		{
			cleanupArray.removeAtAndCopy(i);
		}
	}
}

void HK_CALL hkpConstraintCallbackUtil::fireConstraintAdded( hkpConstraintInstance* constraint ) 
{
	hkSmallArray<hkpConstraintListener*> &listen = constraint->m_listeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("conAddCb", constraint);
			listen[i]->constraintAddedCallback( constraint );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpConstraintCallbackUtil_cleanupNullPointers( cleanupArray );
}

void HK_CALL hkpConstraintCallbackUtil::fireConstraintRemoved( hkpConstraintInstance* constraint ) 
{
	hkSmallArray<hkpConstraintListener*> &listen = constraint->m_listeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("conRemCb", constraint);
			listen[i]->constraintRemovedCallback( constraint );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpConstraintCallbackUtil_cleanupNullPointers( cleanupArray );
}

void HK_CALL hkpConstraintCallbackUtil::fireConstraintDeleted( hkpConstraintInstance* constraint ) 
{
	hkSmallArray<hkpConstraintListener*> &listen = constraint->m_listeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("conDeletedCb", constraint);
			listen[i]->constraintDeletedCallback( constraint );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpConstraintCallbackUtil_cleanupNullPointers( cleanupArray );
}

void HK_CALL hkpConstraintCallbackUtil::fireConstraintBroken( const hkpConstraintBrokenEvent& event ) 
{
	hkSmallArray<hkpConstraintListener*> &listen = event.m_constraintInstance->m_listeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("conBrokenCb", constraint);
			listen[i]->constraintBreakingCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpConstraintCallbackUtil_cleanupNullPointers( cleanupArray );
}

void HK_CALL hkpConstraintCallbackUtil::fireConstraintRepaired( const hkpConstraintRepairedEvent& event ) 
{
	hkSmallArray<hkpConstraintListener*> &listen = event.m_constraintInstance->m_listeners;
	for ( int i = listen.getSize()-1; i >= 0; i-- )
	{
		if (listen[i] != HK_NULL)
		{
			HK_TIME_CODE_BLOCK("conRepairedCb", constraint);
			listen[i]->constraintRepairedCallback( event );
		}
	}
	hkSmallArray<void*>& cleanupArray = reinterpret_cast<hkSmallArray<void*>&>(listen);
	hkpConstraintCallbackUtil_cleanupNullPointers( cleanupArray );
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
