/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONSTRAINT_FORCE_EVENT_CREATOR_H
#define HKNP_CONSTRAINT_FORCE_EVENT_CREATOR_H

#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>


/// Modifier which raises constraint force events from the solver.
/// \sa hknpConstraintForceEvent
class hknpConstraintForceEventCreator : public hknpModifier
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConstraintForceEventCreator );

		//
		// hknpModifier implementation
		//

		HK_FORCE_INLINE int getEnabledFunctions()
		{
			return (1<<FUNCTION_POST_CONSTRAINT_EXPORT);
		}

		void postConstraintExport(
			const hknpSimulationThreadContext& tl, hknpConstraint* constraint,
			hknpImmediateConstraintId immId, hkUint8 constraintAtomFlags, void* runtime ) HK_OVERRIDE;
};


#endif	//HKNP_CONSTRAINT_FORCE_EVENT_CREATOR_H

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
