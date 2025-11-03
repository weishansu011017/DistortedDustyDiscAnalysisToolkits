module BackendExtra

    using PhantomRevealer  
    using GLMakie         

    function activate_backend()
        GLMakie.activate!()
    end

    function open_GLscreen()
        return GLMakie.Screen()
    end

    function close_Fig!(Fax :: FigureAxes)
        GLMakie.close(Fax.screen)
    end
    export activate_backend, close_Fig!
end  