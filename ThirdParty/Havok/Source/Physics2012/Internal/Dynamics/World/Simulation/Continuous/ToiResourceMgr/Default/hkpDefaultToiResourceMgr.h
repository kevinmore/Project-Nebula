/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS2_DEFAULT_TOI_LISTENER_H
#define HK_DYNAMICS2_DEFAULT_TOI_LISTENER_H

#include <Physics2012/Internal/Dynamics/World/Simulation/Continuous/ToiResourceMgr/hkpToiResourceMgr.h>


class hkpWorld;
class hkStepInfo;

extern const hkClass hkpDefaultToiResourceMgrClass;


class hkpDefaultToiResourceMgr : public hkpToiResourceMgr
{
	public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		hkpDefaultToiResourceMgr();
		~hkpDefaultToiResourceMgr();


			/// Return HK_SUCCESS when the passed TOI event should be processed, failure otherwise (don't need to call endToi then).
		virtual hkResult beginToiAndSetupResources(const hkpToiEvent& event, const hkArray<hkpToiEvent>& otherEvents, hkpToiResources& resourcesOut );


			// hkpToiResourceMgr interface implementations
		virtual hkpToiResourceMgrResponse cannotSolve(hkArray<ConstraintViolationInfo>& violatedConstraints);


			// hkpToiResourceMgr interface implementations
		virtual hkpToiResourceMgrResponse resourcesDepleted();


			// hkpToiResourceMgr interface implementations
		virtual void endToiAndFreeResources(const hkpToiEvent& event, const hkArray<hkpToiEvent>& otherEvents, const hkpToiResources& resources );

			// hkReferencedObject implementation
		virtual const hkClass* getClassType() const { return &hkpDefaultToiResourceMgrClass; }

		virtual int getScratchpadCapacity();

	private:
		hkBool shouldHandleGivenToi( const hkpToiEvent& event );
		int m_scratchPadCapacity;

	public:
		int m_defaultScratchpadSize;

			/// The default priority class map, which has three classes.
		static const hkUint8 s_priorityClassMap[];
			/// The default ratios.
		static const hkReal s_priorityClassRatios[];
};


#endif // HK_DYNAMICS2_DEFAULT_TOI_LISTENER_H

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
