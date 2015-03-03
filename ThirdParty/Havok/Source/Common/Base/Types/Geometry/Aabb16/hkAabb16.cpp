/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Aabb16/hkAabb16.h>




void hkAabb16::setExtents( const hkAabb16* aabbsIn, int numAabbsIn )
{
#if defined(HK_USING_GENERIC_INT_VECTOR_IMPLEMENTATION)
	{
		hkAabb16 aabbOut;	aabbOut.setEmpty();
		for(int i=0; i<numAabbsIn; i++)
		{
			aabbOut.includeAabb( aabbsIn[i] );
		}
		this[0] = aabbOut;
	}
#else
	hkIntVector vmin; vmin.load<4>( (const hkUint32*)aabbsIn );
	hkIntVector vmax = vmin;
	for (int i = 1; i < numAabbsIn; i++ )
	{
		hkIntVector a; a.load<4>( (const hkUint32*)&aabbsIn[i]  );
		vmin.setMinS16( vmin, a );
		vmax.setMaxS16( vmax, a );
	}
	vmax.store<4>( (hkUint32*)this);
	vmin.store<2>( (hkUint32*)this);
#endif
}



void hkAabb16::setExtentsOfCenters( const hkAabb16* aabbsIn, int numAabbsIn )
{
#if defined(HK_USING_GENERIC_INT_VECTOR_IMPLEMENTATION)
	{
		hkAabb16 aabbOut;	aabbOut.setEmpty();
		for(int ie=0; ie<numAabbsIn; ie++)
		{
			hkUint16 center[3];
			aabbsIn[ie].getCenter( center );
			aabbOut.includePoint( center );
		}
		this[0] = aabbOut;
	}
#else
	hkIntVector vmin; 
	{
		hkIntVector mi; mi.load<4>( (const hkUint32*)&aabbsIn[0]  );
		hkIntVector ma; ma.setPermutation<hkVectorPermutation::ZWWW>(mi);
		hkIntVector center2; center2.setAddSaturateU16( mi, ma );
		vmin.setShiftRight16<1>( center2);
	}
	hkIntVector vmax = vmin;
	for (int i = 1; i < numAabbsIn; i++ )
	{
		hkIntVector mi; mi.load<4>( (const hkUint32*)&aabbsIn[i]  );
		hkIntVector ma; ma.setPermutation<hkVectorPermutation::ZWWW>(mi);
		hkIntVector center2; center2.setAddSaturateU16( mi, ma );
		hkIntVector center; center.setShiftRight16<1>( center2);

		vmin.setMinS16( vmin, center );
		vmax.setMaxS16( vmax, center );
	}
	vmin.store<2>( ((hkUint32*)this) + 0 );
	vmax.store<2>( ((hkUint32*)this) + 2);
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
