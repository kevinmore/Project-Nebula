/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */



#ifndef HK_DYNAMICS2_DEFAULT_WORLD_MAINTENANCE_MGR_H
#define HK_DYNAMICS2_DEFAULT_WORLD_MAINTENANCE_MGR_H

#include <Physics2012/Dynamics/World/Maintenance/hkpWorldMaintenanceMgr.h>

/// Class hkpDefaultWorldMaintenanceMgr
class hkpDefaultWorldMaintenanceMgr : public hkpWorldMaintenanceMgr
{
	public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
		hkpDefaultWorldMaintenanceMgr();

		virtual void init( hkpWorld* world);
			/// ###ACCESS_CHECKS###( [world,HK_ACCESS_RW] );
		virtual void performMaintenance( hkpWorld* world, hkStepInfo& stepInfo );

			// do all maintenance but ignore island split checks. This is used for multithreaded simulation, where
			// the split check is done in parallel to the solve job
			/// ###ACCESS_CHECKS###( [world,HK_ACCESS_RW] );
		virtual void performMaintenanceNoSplit( hkpWorld* world, hkStepInfo& stepInfo );

	private:
			/// ###ACCESS_CHECKS###( [world,HK_ACCESS_RW] );
		inline void resetWorldTime( hkpWorld* world, hkStepInfo& stepInfo);

			// this is the old Havok400 style deactivation checks at the beginning of each frame
			/// ###ACCESS_CHECKS###( [world,HK_ACCESS_RW] );
		inline void markIslandsForDeactivationDeprecated( hkpWorld* world, hkStepInfo& stepInfo);


	protected:
		/// A range within the time variable will be held
		hkReal m_minAllowedTimeValue;
		hkReal m_maxAllowedTimeValue;
};




#endif // HK_DYNAMICS2_DEFAULT_WORLD_MAINTENANCE_MGR_H

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
