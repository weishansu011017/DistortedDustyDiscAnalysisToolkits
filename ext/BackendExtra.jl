module BackendExtra

    using PhantomRevealer  
    using GLMakie         

    function activate_backend()
        GLMakie.activate!()
    end

    function close_Fig!(Fax :: FigureAxes)
        GLMakie.close(Fax.screen)
    end
    export activate_backend, close_Fig!
end  