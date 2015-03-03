/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_QUALITY_LIBRARY_H
#define HKNP_BODY_QUALITY_LIBRARY_H

#include <Common/Base/Types/hkSignalSlots.h>

#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.h>


///
struct hknpBodyQualityLibraryCinfo
{
	hknpBodyQualityLibraryCinfo()
	{
		m_unitScale = 1.0f;
		m_useWeldingForDefaultObjects = false;
		m_useWeldingForCriticalObjects = false;
	}

	hkReal m_unitScale;
	hkBool m_useWeldingForDefaultObjects;
	hkBool m_useWeldingForCriticalObjects;
};


/// A library of body qualities, indexed by hknpBodyQualityId.
class hknpBodyQualityLibrary : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Empty constructor.
		hknpBodyQualityLibrary();

		/// Serialization constructor.
		hknpBodyQualityLibrary( hkFinishLoadedObjectFlag flag );

		/// Initialize all qualities, including presets.
		void initialize( const hknpBodyQualityLibraryCinfo& cinfo = hknpBodyQualityLibraryCinfo() );

		/// Get read-only access to the qualities buffer.
		HK_FORCE_INLINE const hknpBodyQuality* getBuffer() const;

		/// Get the maximum number of qualities.
		HK_FORCE_INLINE int getCapacity() const;

		/// Get read-only access to a quality.
		HK_FORCE_INLINE const hknpBodyQuality& getEntry( hknpBodyQualityId id ) const;

		/// Update a quality.
		HK_FORCE_INLINE void updateEntry( hknpBodyQualityId id, const hknpBodyQuality& quality );

	public:

		/// Signal fired when a quality is modified in updateEntry()
		HK_DECLARE_SIGNAL( QualityModifiedSignal, hkSignal1<hknpBodyQualityId> );
		QualityModifiedSignal m_qualityModifiedSignal;	//+nosave +overridetype(void*)

	protected:

		/// Qualities array, aligned for SPU access.
		HK_ALIGN16(hknpBodyQuality m_qualities[ hknpBodyQualityId::MAX_NUM_QUALITIES ]);
};

#include <Physics/Physics/Dynamics/Body/hknpBodyQualityLibrary.inl>


#endif // HKNP_BODY_QUALITY_LIBRARY_H

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
