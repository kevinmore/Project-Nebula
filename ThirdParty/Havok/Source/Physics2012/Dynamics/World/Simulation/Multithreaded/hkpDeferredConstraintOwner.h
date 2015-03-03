/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DEFERRED_CONSTRAINT_OWNER_H
#define HK_DEFERRED_CONSTRAINT_OWNER_H

#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintOwner.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>

class hkpDeferredConstraintOwner: public hkpConstraintOwner
{
	public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		enum { FIXED_QUEUE_LENGTH = hkpPhysicsCommandQueue::BYTES_PER_COMMAND * hkpAgentBaseJob::MAX_AGENT_NN_ENTRIES_PER_TASK };

	public:
		HK_FORCE_INLINE hkpDeferredConstraintOwner()
		{
			m_constraintInfo.clear();
			m_constraintAddRemoveCounter = 0;
			m_callbackRequestForAddConstraint = 0; 
		}

		virtual void addConstraintToCriticalLockedIsland( hkpConstraintInstance* constraint )
		{
#if defined (HK_PLATFORM_SPU)
			m_constraintForCommand = m_constraintOnPpu;
#else
			m_constraintForCommand = constraint;
#endif

			m_constraintAddRemoveCounter = m_constraintAddRemoveCounter + 1;
		}

		virtual void removeConstraintFromCriticalLockedIsland( hkpConstraintInstance* constraint )
		{
#if defined (HK_PLATFORM_SPU)
			m_constraintForCommand = m_constraintOnPpu;
#else
			m_constraintForCommand = constraint;
#endif

			m_constraintAddRemoveCounter = m_constraintAddRemoveCounter - 1;
		}

		void addCallbackRequest( hkpConstraintInstance* constraint, int request )
		{
			if ( constraint->m_internal )
			{
				constraint->m_internal->m_callbackRequest |= request;
			}
			else
			{
				m_callbackRequestForAddConstraint = m_callbackRequestForAddConstraint | request;
			}
		}


		hkPadSpu<int>	m_constraintAddRemoveCounter;

		hkPadSpu<hkpConstraintInstance*> m_constraintForCommand;
#if defined (HK_PLATFORM_HAS_SPU)
		hkPadSpu<hkpConstraintInstance*> m_constraintOnPpu;
#endif
		hkPadSpu<int> m_callbackRequestForAddConstraint;

		hkpFixedSizePhysicsCommandQueue<FIXED_QUEUE_LENGTH> m_commandQueue;
};
#endif // HK_DEFERRED_CONSTRAINT_OWNER_H

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
