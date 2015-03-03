/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_MANAGER_H
#define HKNP_SHAPE_MANAGER_H


/// A helper class to track shapes that are in use by a world.
class hknpShapeManager
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeManager );

		/// Constructor.
		hknpShapeManager();

		/// Destructor.
		~hknpShapeManager();

		/// Register a body which has a mutable shape.
		/// The shape will signal this manager whenever it is mutated.
		void registerBodyWithMutableShape( hknpBody& body );

		/// Deregister a body which was previously registered.
		void deregisterBodyWithMutableShape( hknpBody& body );

		/// Check whether any known shapes were mutated.
		HK_FORCE_INLINE hkBool isAnyShapeMutated() const { return m_isAnyShapeMutated; }

		/// Update stale engine data for any shapes which were mutated since the last time this method was called.
		void processMutatedShapes( hknpWorld* world );

	protected:

		/// Information about a mutable shape
		struct MutableShapeInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeManager::MutableShapeInfo );

			MutableShapeInfo( hknpShapeManager *shapeManager );
			~MutableShapeInfo();

			void init( const hknpShape* shape );
			void deinit();

			// Signal handlers
			void onShapeMutated( hkUint8 mutationFlags );
			void onShapeDestroyed();

			hknpShapeManager* m_shapeManager;		///< the owning shape manager
			const hknpShape* m_shape;				///< the mutable shape
			hkArray<hknpBodyId> m_bodyIds;			///< the bodies currently using the mutable shape
			hknpShape::MutationFlags m_mutations;	///< the mutations that were done
		};

	protected:

		/// Active mutable shape infos.
		hkArray<MutableShapeInfo*> m_mutableShapeInfos;

		/// Free mutable shape infos, available for reuse.
		hkArray<MutableShapeInfo*> m_freeMutableShapeInfos;

		/// Set to true if any shape is mutated.
		hkBool m_isAnyShapeMutated;
};

#endif // HKNP_SHAPE_MANAGER_H

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
